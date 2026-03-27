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
        keygen.create_galois_keys(gk_);
        
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
     * OPÉRATION RÉELLE : Produit Matrice-Vecteur Naïf (Fidèle au modèle)
     * On multiplie un vecteur d'input (Ciphertext) par une matrice (Plaintext Matrix)
     * sans les optimisations MOAI pour montrer la différence.
     */
    py::bytes process_layer_compact(py::bytes enc_input, py::list matrix_plain) {
        // Charger l'entrée chiffrée (Vecteur X)
        string in_str = enc_input;
        stringstream ss_in(in_str);
        Ciphertext ct_in;
        ct_in.load(*context_, ss_in);

        Ciphertext ct_res;
        bool first = true;

        // On parcourt chaque ligne de la matrice (256 lignes)
        // C'est l'approche naïve sans Packing Diagonal
        for (size_t i = 0; i < matrix_plain.size(); i++) {
            py::list row = matrix_plain[i].cast<py::list>();
            vector<double> row_vec;
            for (auto item : row) row_vec.push_back(item.cast<double>());

            Plaintext pt_row;
            encoder_->encode(row_vec, scale_, pt_row);

            // Rotation du vecteur d'entrée pour aligner les éléments (Calcul partiel)
            Ciphertext ct_rot;
            evaluator_->rotate_vector(ct_in, (int)i, gk_, ct_rot);
            
            // Multiplication
            evaluator_->multiply_plain_inplace(ct_rot, pt_row);
            
            if (first) {
                ct_res = ct_rot;
                first = false;
            } else {
                evaluator_->add_inplace(ct_res, ct_rot);
            }
        }
        
        // Relinearization et Rescale
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
    GaloisKeys gk_;
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
