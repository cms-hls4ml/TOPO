#include "NN/topo_v1.h" //include of the top level of HLS model
#include "emulator.h" //include of emulator modeling
#include <any>
#include "ap_fixed.h"

using namespace hls4ml_topo_v1;

class topo_emulator_v1 : public hls4mlEmulator::Model {

private:
    input_t _input[N_INPUT_1_1];
    result_t _result[N_LAYER_11];

public: 
    virtual void prepare_input(std::any input) {
        input_t* input_p = std::any_cast<input_t*>(input);
        for(int i = 0; i < N_INPUT_1_1; ++i)
            _input[i] = std::any_cast<input_t>(input_p[i]);
    }

    virtual void predict() {
        topo_v1(_input, _result);
    }
  
    virtual void read_result(std::any result) {
        result_t *result_p = std::any_cast<result_t*>(result);
        for (int i = 0; i < N_LAYER_11; ++i)
            result_p[i] = _result[i];
    }
};

extern "C" hls4mlEmulator::Model* create_model() {
    return new topo_emulator_v1;
}

extern "C" void destroy_model(hls4mlEmulator::Model* m) {
    delete m;
}
