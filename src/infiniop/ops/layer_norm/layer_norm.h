#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__

#include "../../operator.h"
#include "../../tensor.h"
#include "infinicore.h"

namespace op::layer_norm {

struct LayerNormInfo {
    size_t _batch_size;         
    size_t _normalized_size;    
    size_t total_elements;     
    size_t input_size;
    size_t output_size;
    
    infiniDtype_t dtype;
    infiniDtype_t wtype;        
    infiniDtype_t btype;        
    infiniDtype_t atype;        
    float eps;                  
    float epsilon;              
    bool has_bias;              
    
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> bias_shape;
    std::vector<size_t> input_std_deviation_shape;
    std::vector<size_t> input_standardization_shape;
    std::vector<size_t> shape; 
    
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> bias_strides;
    std::vector<ptrdiff_t> input_std_deviation_strides;
    std::vector<ptrdiff_t> input_standardization_strides;
    
    size_t ndim() const { return shape.size(); }
    size_t dim() const { return _normalized_size; }
    size_t batch_size() const {
        return _batch_size;
    }
};

} // namespace op::layer_norm

#endif // __LAYER_NORM_H__