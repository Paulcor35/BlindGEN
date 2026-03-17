#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "seal/seal.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace py = pybind11;
using namespace seal;
using namespace std;

/**
 * Moteur BlindGEN (MÉTHODE COMPACT) - VERSION RÉELLE MICROSOFT SEAL
 * ---------------------------------------------------------------
 */

class BlindEngineReal {
public:
    BlindEngineReal(size_t poly_degree, double scale_val) {
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(poly_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_degree, { 60, 40, 40, 60 }));
        
        context_ = make_shared<SEALContext>(parms);
        KeyGenerator keygen(*context_);
        
        sk_ = keygen.secret_key();
        keygen.create_public_key(pk_);
        keygen.create_relin_keys(rk_);
        
        encoder_ = make_unique<CKKSEncoder>(*context_);
        evaluator_ = make_unique<Evaluator>(*context_);
        encryptor_ = make_unique<Encryptor>(*context_, pk_);
        scale_ = scale_val;
        decryptor_ = make_unique<Decryptor>(*context_, sk_);
        cout << "[C++ SEAL] Moteur Microsoft SEAL (AVEUGLE) initialisé." << endl;
    }

    /**
     * CHIFFREMENT CLIENT-SIDE (Outil pour la démo)
     */
    py::bytes encrypt_data(py::array_t<double> input) {
        auto in_buf = input.request();
        vector<double> data((double*)in_buf.ptr, (double*)in_buf.ptr + in_buf.size);
        
        Plaintext pt;
        encoder_->encode(data, scale_, pt);
        Ciphertext ct;
        encryptor_->encrypt(pt, ct);
        
        stringstream ss;
        ct.save(ss);
        return py::bytes(ss.str());
    }

    /**
     * OPÉRATION SOUVERAINE : Multiplie deux Ciphertexts (Données x Poids)
     * Le serveur ne connaît ni l'un ni l'autre.
     */
    py::bytes process_layer_compact(py::bytes enc_input, py::bytes enc_weights) {
        // Charger l'entrée chiffrée
        string in_str = enc_input;
        stringstream ss_in(in_str);
        Ciphertext ct_in;
        ct_in.load(*context_, ss_in);

        // Charger les poids chiffrés
        string w_str = enc_weights;
        stringstream ss_w(w_str);
        Ciphertext ct_w;
        ct_w.load(*context_, ss_w);
        
        // MULTIPLICATION HOMOMORPHIQUE (Ciphertext * Ciphertext)
        Ciphertext ct_res;
        evaluator_->multiply(ct_in, ct_w, ct_res);
        
        // Relinearization (Obligatoire après CT*CT)
        evaluator_->relinearize_inplace(ct_res, rk_);
        evaluator_->rescale_to_next_inplace(ct_res);

        stringstream ss_res;
        ct_res.save(ss_res);
        return py::bytes(ss_res.str());
    }

    /**
     * DÉCHIFFREMENT CLIENT-SIDE
     */
    py::list decrypt_result(py::bytes encrypted_data) {
        string data = encrypted_data;
        stringstream ss(data);
        Ciphertext ct;
        ct.load(*context_, ss);

        Plaintext pt;
        decryptor_->decrypt(ct, pt);
        
        vector<double> result;
        encoder_->decode(pt, result);
        
        py::list out;
        for(double d : result) out.append(d);
        return out;
    }

private:
    shared_ptr<SEALContext> context_;
    PublicKey pk_;
    SecretKey sk_;
    RelinKeys rk_;
    unique_ptr<CKKSEncoder> encoder_;
    unique_ptr<Evaluator> evaluator_;
    unique_ptr<Encryptor> encryptor_;
    unique_ptr<Decryptor> decryptor_;
    double scale_;
};

PYBIND11_MODULE(blind_engine_sov, m) {
    m.doc() = "Moteur BlindGEN Full-FHE (Microsoft SEAL)";

    py::class_<BlindEngineReal>(m, "BlindEngine")
        .def(py::init<size_t, double>())
        .def("encrypt_data", &BlindEngineReal::encrypt_data)
        .def("process_layer_compact", &BlindEngineReal::process_layer_compact)
        .def("decrypt_result", &BlindEngineReal::decrypt_result);
}
