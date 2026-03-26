#include <omp.h>
#include <seal/seal.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdint>
#include <exception>

using namespace std;
using namespace seal;

struct LayerWeights {
    int rows, cols;
    vector<double> W;
    vector<double> B;
};

bool load_weights(const string& path, LayerWeights& lw) {
    ifstream f(path, ios::binary);
    if (!f.is_open()) return false;
    f.read((char*)&lw.rows, sizeof(int));
    f.read((char*)&lw.cols, sizeof(int));
    lw.W.resize((size_t)lw.rows * lw.cols);
    f.read((char*)lw.W.data(), lw.W.size() * sizeof(double));
    int b_size = 0;
    f.read((char*)&b_size, sizeof(int));
    lw.B.resize(b_size);
    f.read((char*)lw.B.data(), lw.B.size() * sizeof(double));
    return true;
}

int main() {
    try {
        cout << "--- MOAI NATIVE BENCHMARK ---" << endl;

        LayerWeights lw;
        if (!load_weights("../../../gpt2_weights.bin", lw)) {
            if (!load_weights("gpt2_weights.bin", lw)) {
                cerr << "Erreur : gpt2_weights.bin non trouve" << endl;
                return 1;
            }
        }

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(16384);
        parms.set_coeff_modulus(CoeffModulus::Create(16384, {60, 46, 46, 46, 60}));
        
        SEALContext context(parms);
        KeyGenerator keygen(context);
        
        cout << "Generation Galois Keys (TOUTES)..." << endl;
        GaloisKeys gal_keys;
        // On genere tout pour eviter les erreurs de rotation steps invalides
        keygen.create_galois_keys(gal_keys); 
        cout << "Galois Keys OK." << endl;

        PublicKey pk;
        keygen.create_public_key(pk);
        Encryptor encryptor(context, pk);
        Evaluator evaluator(context);
        CKKSEncoder encoder(context);
        double scale = pow(2.0, 46);

        // Setup Matrix
        int n = 256;
        int n1 = (int)ceil(sqrt(n));
        int n2 = (int)ceil((double)n / n1);
        vector<vector<Plaintext>> cached_P_ij(n2, vector<Plaintext>(n1));
        size_t slot_count = encoder.slot_count();

        cout << "Setup BSGS Poids (Encodage Plaintexts)..." << endl;
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n1; j++) {
                int k = i * n1 + j;
                if (k < n) {
                    vector<double> p_array(slot_count, 0.0);
                    for (size_t y = 0; y < slot_count; y++) {
                        long long mapped = ((long long)y - (long long)i * n1) % n;
                        if (mapped < 0) mapped += n; 
                        p_array[y] = lw.W[(size_t)mapped * lw.cols + ((mapped + k) % n)];
                    }
                    encoder.encode(p_array, context.first_parms_id(), scale, cached_P_ij[i][j]);
                }
            }
        }

        // Simu Input
        vector<double> h_input(n, 1.0);
        Plaintext pt_v;
        encoder.encode(h_input, scale, pt_v);
        Ciphertext ct_v;
        encryptor.encrypt(pt_v, ct_v);

        cout << "Inference boucle native active..." << endl;
        auto start_bench = chrono::high_resolution_clock::now();
        
        for(int t=0; t<10; t++) {
            auto s = chrono::high_resolution_clock::now();
            
            // BSGS Matmul
            vector<Ciphertext> V(n1);
            V[0] = ct_v;
            for (int j = 1; j < n1; j++) evaluator.rotate_vector(V[j-1], 1, gal_keys, V[j]);

            vector<Ciphertext> I(n2);
            #pragma omp parallel for
            for (int i = 0; i < n2; i++) {
                Ciphertext acc; bool init = false;
                for (int j = 0; j < n1; j++) {
                    int k = i * n1 + j; if (k >= n) break;
                    Ciphertext p; evaluator.multiply_plain(V[j], cached_P_ij[i][j], p);
                    if (!init) { acc = p; init = true; } else evaluator.add_inplace(acc, p);
                }
                I[i] = acc;
            }

            Ciphertext Y; bool Y_inited = false;
            for (int i = n2 - 1; i >= 0; i--) {
                if (!Y_inited) { Y = I[i]; Y_inited = true; }
                else { evaluator.rotate_vector_inplace(Y, n1, gal_keys); Y.scale() = I[i].scale(); evaluator.add_inplace(Y, I[i]); }
            }
            evaluator.rescale_to_next_inplace(Y);
            Y.scale() = scale;
            ct_v = Y;

            auto e = chrono::high_resolution_clock::now();
            double d = chrono::duration<double>(e - s).count();
            printf("Native Token %d : %.2f tok/s\n", t+1, 1.0/d);
            fflush(stdout);
        }
        
        auto end_bench = chrono::high_resolution_clock::now();
        double total_d = chrono::duration<double>(end_bench - start_bench).count();
        cout << "---" << endl;
        cout << "MOYENNE NATIVE FINALE : " << 10.0 / total_d << " tok/s" << endl;

    } catch (const exception &e) {
        cerr << "CRASH : " << e.what() << endl;
        return 1;
    }
    return 0;
}
