#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <iterator>
#include "xoshiro1024pp.hpp"
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

using namespace std;

const int THREAD_COUNT = thread::hardware_concurrency();

mutex output_mutex;
atomic<bool> found(false);
uint64_t found_seed = 0;

vector<unsigned char> read_file(const string& filename) {
    ifstream file(filename, ios::binary);
    return vector<unsigned char>((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

void write_file(const string& filename, const vector<unsigned char>& data) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

vector<uint64_t> file_to_bits_u64(const vector<unsigned char>& data, size_t& out_bit_count) {
    size_t bit_count = data.size() * 8;
    vector<uint64_t> bits((bit_count + 63) / 64, 0);

    for (size_t i = 0; i < bit_count; ++i) {
        if (data[i / 8] & (1 << (7 - (i % 8)))) {
            bits[i / 64] |= (uint64_t(1) << (63 - (i % 64)));
        }
    }

    if (bit_count % 64 != 0) {
        int valid_bits = bit_count % 64;
        uint64_t mask = ((uint64_t)1 << valid_bits) - 1;
        bits.back() &= (mask << (64 - valid_bits));
    }

    out_bit_count = bit_count;
    return bits;
}

vector<uint64_t> generate_bits_u64(uint64_t seed, size_t bit_count) {
    xoshiro1024pp rng(seed);
    size_t word_count = (bit_count + 63) / 64;
    vector<uint64_t> bits(word_count);

    for (size_t i = 0; i < word_count; ++i) {
        bits[i] = rng.next();
    }

    if (bit_count % 64 != 0) {
        int valid_bits = bit_count % 64;
        uint64_t mask = ((uint64_t)1 << valid_bits) - 1;
        mask <<= (64 - valid_bits);
        bits[word_count - 1] &= mask;
    }

    return bits;
}

vector<unsigned char> bits_to_bytes_u64(const vector<uint64_t>& bits, size_t bit_count) {
    vector<unsigned char> bytes((bit_count + 7) / 8, 0);

    for (size_t i = 0; i < bit_count; ++i) {
        if (bits[i / 64] & (uint64_t(1) << (63 - (i % 64)))) {
            bytes[i / 8] |= (1 << (7 - (i % 8)));
        }
    }
    return bytes;
}

void search_seed(const vector<uint64_t>& target_bits, size_t bit_count, uint64_t start_seed, uint64_t step) {
    uint64_t seed = start_seed;
    while (!found) {
        vector<uint64_t> gen = generate_bits_u64(seed, bit_count);
        if (gen == target_bits) {
            lock_guard<mutex> lock(output_mutex);
            if (!found) {
                found = true;
                found_seed = seed;
            }
            break;
        }
        seed += step;
        if (seed % 1'000'000 == 0) {
            cout << "PROGRESS (" << seed << ")" << endl;
        }
    }
}

string read_kernel(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("ERROR (Failed to open kernel file)");
    return string((istreambuf_iterator<char>(file)),
                       istreambuf_iterator<char>());
}

void compress(const string& input_file, const string& output_file) {
    vector<unsigned char> data = read_file(input_file);
    size_t bit_count;
    vector<uint64_t> target_bits = file_to_bits_u64(data, bit_count);


    size_t word_count = (bit_count + 63) / 64;

    cout << "TARGET > (" << bit_count << " bits)" << endl;
    cout << "WORDS > (" << word_count << ")" << endl;
    cout << "THREADS > (" << THREAD_COUNT << ") (CPU)" << endl;

    vector<thread> threads;
    for (int i = 0; i < THREAD_COUNT; ++i) {
        threads.emplace_back(search_seed, cref(target_bits), bit_count, i, THREAD_COUNT);
    }

    for (auto& t : threads) {
        t.join();
    }

    if (found) {
        ofstream out(output_file, ios::binary);
        out.write(reinterpret_cast<const char*>(&found_seed), sizeof(found_seed));
        out.write(reinterpret_cast<const char*>(&bit_count), sizeof(bit_count));
        out.close();

        cout << "DONE (" << found_seed << ") (" << bit_count << " bits) > (" << output_file << ")" << endl;
    } else {
        cout << "ERROR" << endl;
    }
}

void compress_cl(const string& input_file, const string& output_file, size_t chunk_size = 1'000'000'000) {
    vector<unsigned char> data = read_file(input_file);
    size_t bit_count;
    vector<uint64_t> target_bits = file_to_bits_u64(data, bit_count);

    size_t word_count = (bit_count + 63) / 64;

    cout << "TARGET > (" << bit_count << " bits)" << endl;
    cout << "WORDS > (" << word_count << ")" << endl;
    cout << "CHUNK SIZE > (" << chunk_size << ") (OpenCL)" << endl;


    try {
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw runtime_error("ERROR (No OpenCL platforms found)");

        cl::Platform platform = platforms[0];
        vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) throw runtime_error("ERROR (No OpenCL device found)");

        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // 2. Read and build kernel
        string kernel_source = read_kernel("pakhomov-gsc.cl");
        cl::Program program(context, kernel_source);
        try {
            program.build("-cl-std=CL2.0");
        } catch (...) {
            cerr << "ERROR (Kernel build)\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
            throw;
        }

        cl::Kernel kernel(program, "seed_search");

        // 3. Create buffers
        cl::Buffer buffer_target_bits(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(uint64_t) * target_bits.size(), target_bits.data());
        cl::Buffer buffer_out_seed(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t));
        cl::Buffer buffer_found_flag(context, CL_MEM_READ_WRITE, sizeof(int));

        uint64_t start_seed = 0;
        int found_flag = 0;

        while (true) {
            kernel.setArg(0, buffer_out_seed);
            kernel.setArg(1, buffer_target_bits);
            kernel.setArg(2, (cl_ulong)bit_count);
            kernel.setArg(3, (cl_ulong)start_seed);
            kernel.setArg(4, buffer_found_flag);

            // Reset flag
            found_flag = 0;
            queue.enqueueWriteBuffer(buffer_found_flag, CL_TRUE, 0, sizeof(int), &found_flag);

            // 5. Launch kernel
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(chunk_size));
            queue.finish();

            // 6. Check if a match was found
            queue.enqueueReadBuffer(buffer_found_flag, CL_TRUE, 0, sizeof(int), &found_flag);
            if (found_flag != 0) {
                uint64_t found_seed;
                queue.enqueueReadBuffer(buffer_out_seed, CL_TRUE, 0, sizeof(uint64_t), &found_seed);

                ofstream out(output_file, ios::binary);
                out.write(reinterpret_cast<const char*>(&found_seed), sizeof(found_seed));
                out.write(reinterpret_cast<const char*>(&bit_count), sizeof(bit_count));
                out.close();

                cout << "DONE (" << found_seed << ") (" << bit_count << " bits) > (" << output_file << ")" << endl;
                break;
            }

            start_seed += chunk_size;
            cout << "PROGRESS (" << start_seed << ")" << endl;

        }

    } catch (const exception& e) {
        cerr << "ERROR (OpenCL) > " << e.what() << endl;
    }
}

void decompress(const string& input_file, const string& output_file) {
    ifstream in(input_file, ios::binary);
    uint64_t seed;
    size_t bit_count;
    in.read(reinterpret_cast<char*>(&seed), sizeof(seed));
    in.read(reinterpret_cast<char*>(&bit_count), sizeof(bit_count));

    vector<uint64_t> bits = generate_bits_u64(seed, bit_count);
    vector<unsigned char> bytes = bits_to_bytes_u64(bits, bit_count);
    write_file(output_file, bytes);

    cout << "DONE (" << seed << ") (" << bit_count << " bits) > (" << output_file << ")" << endl;
}

// Main
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Pakhomov GSC (Generative Seed Compression) (2025)\n";
        cout << "USAGE\n";
        cout << "  " << argv[0] << " compress <input_file>\n";
        cout << "  " << argv[0] << " compress-cl <input_file> [chunk_size]\n";
        cout << "  " << argv[0] << " decompress <compressed_file>\n";
        return 1;
    }

    string command = argv[1];
    if (command == "compress") {
        string input = argv[2];
        string output = argv[2] + string(".pgsz");
        compress(input, output);
    } else if (command == "compress-cl") {
        string input = argv[2];
        string output = argv[2] + string(".pgsz");
        size_t chunk_size = (argc > 3) ? stoull(argv[3]) : 1'000'000'000;
        compress_cl(input, output, chunk_size);
    } else if (command == "decompress") {
        string input = argv[2];
        string output = argv[2];

        if (output.size() > 5 && output.rfind(".pgsz") == output.size() - 5) {
            output = output.substr(0, output.size() - 5);
        }
        decompress(input, output);
    } else {
        cout << "ERROR (Command not found) > " << command << endl;
        return 1;
    }

    return 0;
}
