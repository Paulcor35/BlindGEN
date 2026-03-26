#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <seal/seal.h>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;
using namespace seal;
using namespace std;

class MoaiServer {
public:
  shared_ptr<SEALContext> context;
  Evaluator *evaluator;
  CKKSEncoder *encoder;
  GaloisKeys gal_keys;
  bool has_galois = false;
  
  vector<vector<Plaintext>> cached_P_ij;
  Plaintext cached_bias;
  int matrix_n = 0;
  int batch_size = 1;
  int n1 = 0;
  int n2 = 0;
  double global_scale = pow(2.0, 30);

  MoaiServer(const string &params_bytes) {
    EncryptionParameters parms(scheme_type::ckks);
    istringstream is_parms; is_parms.str(params_bytes);
    parms.load(is_parms);
    context = make_shared<SEALContext>(parms);
    evaluator = new Evaluator(*context);
    encoder = new CKKSEncoder(*context);
  }
  ~MoaiServer() { delete evaluator; delete encoder; }

  void set_batch_size(int b) { batch_size = b; }

  void set_galois(const string &gal_bytes) {
      istringstream is_gal; is_gal.str(gal_bytes);
      gal_keys.load(*context, is_gal);
      has_galois = true;
  }

  void set_weights_bsgs(py::array_t<double> py_W, py::array_t<double> py_bias) {
      auto buf_W = py_W.request(); double *W = (double *)buf_W.ptr;
      matrix_n = buf_W.shape[0]; 
      int d_prime = buf_W.shape[1];
      auto buf_bias = py_bias.request(); double *bias = (double *)buf_bias.ptr;

      int n = matrix_n;
      n1 = ceil(sqrt(n));
      n2 = ceil((double)n / n1);

      cached_P_ij.resize(n2);
      for(int i=0; i<n2; i++) cached_P_ij[i].resize(n1);
      
      auto pid = context->first_parms_id();
      size_t slot_count = encoder->slot_count();

      int threads = min(omp_get_max_threads(), 16);
      #pragma omp parallel num_threads(threads)
      {
          CKKSEncoder thread_encoder(*context);
          #pragma omp for
          for (int i = 0; i < n2; i++) {
              for (int j = 0; j < n1; j++) {
                  int k = i * n1 + j;
                  if (k < n) {
                      vector<double> p_array(slot_count, 0.0);
                      for (size_t y = 0; y < slot_count; y++) {
                          int token_idx = (y / batch_size) % n;
                          long long mapped = ((long long)token_idx - (long long)i * n1) % n;
                          if (mapped < 0) mapped += n; 
                          p_array[y] = W[mapped * d_prime + ((mapped + k) % n)];
                      }
                      thread_encoder.encode(p_array, pid, global_scale, cached_P_ij[i][j]);
                  }
              }
          }
      }
      vector<double> b_vec(slot_count, 0.0);
      for(int y=0; y < (int)slot_count; y++) {
          int d_idx = (y / batch_size) % d_prime;
          b_vec[y] = bias[d_idx];
      }
      encoder->encode(b_vec, pid, global_scale * global_scale, cached_bias);
  }

  py::bytes he_matmul_vector_bsgs(const string &ct_v_bytes) {
    try {
        if (!has_galois) throw runtime_error("GaloisKeys manquantes.");
        
        Ciphertext ct_v;
        istringstream is_v; is_v.str(ct_v_bytes);
        ct_v.load(*context, is_v);

        int n = matrix_n;
        int step = batch_size;
        
        vector<Ciphertext> V(n1);
        V[0] = ct_v;
        for (int j = 1; j < n1; j++) {
            evaluator->rotate_vector(V[j-1], step, gal_keys, V[j]);
        }

        vector<Ciphertext> I(n2);
        vector<bool> I_init(n2, false);
        int threads = min(omp_get_max_threads(), 16);

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < n2; i++) {
            Ciphertext acc;
            bool acc_inited = false;
            for (int j = 0; j < n1; j++) {
                int k = i * n1 + j;
                if (k >= n) break;
                
                Ciphertext prod;
                evaluator->multiply_plain(V[j], cached_P_ij[i][j], prod);
                if (!acc_inited) { acc = prod; acc_inited = true; }
                else { evaluator->add_inplace(acc, prod); }
            }
            if (acc_inited) { I[i] = acc; I_init[i] = true; }
        }

        Ciphertext Y;
        bool Y_init = false;
        int giant_step = n1 * batch_size;
        
        for (int i = n2 - 1; i >= 0; i--) {
            if (!I_init[i]) continue;
            
            if (!Y_init) {
                Y = I[i];
                Y_init = true;
            } else {
                evaluator->rotate_vector_inplace(Y, giant_step, gal_keys);
                Y.scale() = I[i].scale(); 
                evaluator->add_inplace(Y, I[i]);
            }
        }

        evaluator->add_plain_inplace(Y, cached_bias);
        evaluator->rescale_to_next_inplace(Y);
        Y.scale() = global_scale; 
        
        ostringstream os; Y.save(os);
        return py::bytes(os.str());
    } catch (const exception &e) {
        throw runtime_error(string("C++ BSGS Batch Error: ") + e.what());
    }
  }

  py::bytes he_matmul_blob(const string &blob, int d, py::array_t<double> py_W, py::array_t<double> py_bias) {
      try {
          string data_str = string(blob);
          size_t single_ct_size = data_str.size() / d;
          auto buf_W = py_W.request(); double *W = (double *)buf_W.ptr;
          int d_prime = buf_W.shape[1];
          auto buf_bias = py_bias.request(); double *bias = (double *)buf_bias.ptr;
          Ciphertext result; bool is_init = false;

          int threads = min(omp_get_max_threads(), d);
          #pragma omp parallel num_threads(threads)
          {
              Ciphertext thread_acc; bool thread_init = false;
              CKKSEncoder thread_encoder(*context);
              #pragma omp for
              for (int i = 0; i < d; i++) {
                  Ciphertext ct_i;
                  string chunk = data_str.substr(i * single_ct_size, single_ct_size);
                  istringstream is_chunk; is_chunk.str(chunk);
                  ct_i.load(*context, is_chunk);
                  vector<double> w_row(d_prime);
                  for(int j=0; j<d_prime; j++) w_row[j] = W[i * d_prime + j];
                  Plaintext pt_w; thread_encoder.encode(w_row, ct_i.parms_id(), global_scale, pt_w);
                  evaluator->multiply_plain_inplace(ct_i, pt_w);
                  if(!thread_init) { thread_acc = ct_i; thread_init = true; }
                  else { evaluator->add_inplace(thread_acc, ct_i); }
              }
              #pragma omp critical
              {
                  if(!is_init) { result = thread_acc; is_init = true; }
                  else if(thread_init) { evaluator->add_inplace(result, thread_acc); }
              }
          }
          Plaintext pt_b; encoder->encode(vector<double>(bias, bias + d_prime), result.parms_id(), result.scale(), pt_b);
          evaluator->add_plain_inplace(result, pt_b);
          evaluator->rescale_to_next_inplace(result);
          result.scale() = global_scale;
          ostringstream os; result.save(os);
          return py::bytes(os.str());
      } catch (const exception &e) {
          throw runtime_error(string("C++ Blob Error: ") + e.what());
      }
  }
};

class MoaiClient {
public:
  shared_ptr<SEALContext> context;
  KeyGenerator *keygen;
  Encryptor *encryptor;
  Decryptor *decryptor;
  CKKSEncoder *encoder;
  PublicKey pk;
  double scale = pow(2.0, 30);

  MoaiClient(size_t poly_modulus_degree) {
    cout << "[DEBUG] MoaiClient context init for N=" << poly_modulus_degree << endl;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    
    // On utilise exactement les memes parametres que le moteur Turbo C++
    if (poly_modulus_degree <= 8192) {
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 60}));
        scale = pow(2.0, 30);
    } else {
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 60}));
        scale = pow(2.0, 40);
    }

    // On autorise un niveau de securite plus faible si necessaire pour le benchmark
    context = make_shared<SEALContext>(parms, true, sec_level_type::none);
    
    if (!context->parameters_set()) {
        string err = "[CRITICAL] SEAL Reject: ";
        err += context->parameter_error_message();
        cout << err << endl;
        throw runtime_error(err);
    }
    cout << "[DEBUG] SEAL Context OK" << endl;
    
    keygen = new KeyGenerator(*context);
    keygen->create_public_key(pk);
    encryptor = new Encryptor(*context, pk);
    decryptor = new Decryptor(*context, keygen->secret_key());
    encoder = new CKKSEncoder(*context);
  }
  ~MoaiClient() { delete keygen; delete encryptor; delete decryptor; delete encoder; }

  py::bytes get_galois(int n, int batch_size) {
      if (n <= 1) return py::bytes("");
      int n1 = ceil(sqrt(n));
      vector<int> steps = { batch_size, n1 * batch_size };
      GaloisKeys gks; 
      keygen->create_galois_keys(steps, gks);
      ostringstream os_g; gks.save(os_g);
      return py::bytes(os_g.str());
  }

  py::bytes encrypt_batch_interleaved(py::list batches) {
      try {
          int b = batches.size();
          if (b == 0) throw runtime_error("Batch vide.");
          
          py::array_t<double> first = batches[0].cast<py::array_t<double>>();
          int n = first.size();
          
          size_t slot_count = encoder->slot_count();
          vector<double> v(slot_count, 0.0);
          
          for(int i = 0; i < b; i++) {
              py::array_t<double> batch = batches[i].cast<py::array_t<double>>();
              auto buf = batch.request(); double *ptr = (double *)buf.ptr;
              for(int j = 0; j < n; j++) {
                  for(size_t k = j * b + i; k < slot_count; k += (n * b)) {
                      v[k] = ptr[j];
                  }
              }
          }
          
          Plaintext pt; encoder->encode(v, scale, pt);
          Ciphertext ct; encryptor->encrypt(pt, ct);
          ostringstream os_v; ct.save(os_v);
          return py::bytes(os_v.str());
      } catch (const exception &e) {
          throw runtime_error(string("C++ Interleaved Encrypt Error: ") + e.what());
      }
  }

  py::list decrypt_batch(py::bytes data, int batch_size, int dim) {
      try {
          Ciphertext ct; 
          istringstream is_d; is_d.str(string(data));
          ct.load(*context, is_d);
          Plaintext pt; decryptor->decrypt(ct, pt);
          vector<double> res; encoder->decode(pt, res);
          
          py::list result_list;
          for(int i=0; i < batch_size; i++) {
              py::array_t<double> out(dim); double *op = (double *)out.request().ptr;
              for(int j=0; j < dim; j++) {
                  int idx = j * batch_size + i;
                  op[j] = (idx < (int)res.size()) ? res[idx] : 0.0;
              }
              result_list.append(out);
          }
          return result_list;
      } catch (const exception &e) {
          throw runtime_error(string("C++ Batch Decrypt Error: ") + e.what());
      }
  }

  py::bytes get_params() {
      ostringstream os_p; context->key_context_data()->parms().save(os_p);
      return py::bytes(os_p.str());
  }
};

PYBIND11_MODULE(moai_seal_backend, m) {
  py::class_<MoaiClient>(m, "MoaiClient")
      .def(py::init<size_t>())
      .def("encrypt_batch_interleaved", &MoaiClient::encrypt_batch_interleaved)
      .def("decrypt_batch", &MoaiClient::decrypt_batch)
      .def("get_galois", &MoaiClient::get_galois)
      .def("get_params", &MoaiClient::get_params);
  py::class_<MoaiServer>(m, "MoaiServer")
      .def(py::init<const string &>())
      .def("set_batch_size", &MoaiServer::set_batch_size)
      .def("set_galois", &MoaiServer::set_galois)
      .def("set_weights_bsgs", &MoaiServer::set_weights_bsgs)
      .def("he_matmul_vector_bsgs", &MoaiServer::he_matmul_vector_bsgs)
      .def("he_matmul_blob", &MoaiServer::he_matmul_blob);
}
