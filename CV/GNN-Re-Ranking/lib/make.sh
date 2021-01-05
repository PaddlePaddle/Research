include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

CUDNN_dir=YOUR_CUDNN_DIR
CUDA_dir=YOUR_CUDA_DIR

echo $include_dir
echo $lib_dir

# PaddlePaddel >=1.6.1, 仅需要include ${include_dir} 和 ${include_dir}/third_party
nvcc vmat.cu -c -o vmat.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${CUDNN_dir} \
    -I ${include_dir}/third_party \

nvcc qe.cu -c -o qe.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${CUDNN_dir} \
    -I ${include_dir}/third_party \

g++ vmat.cc vmat.cu.o qe.cc qe.cu.o -o rerank.so -shared -fPIC -std=c++11 -O3 -DPADDLE_WITH_MKLDNN \
  -I ${include_dir} \
  -I ${include_dir}/third_party \
  -L ${CUDA_dir} \
  -L ${lib_dir} -lpaddle_framework -lcudart
