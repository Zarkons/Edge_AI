# Tenstorrent mini example

This is only a mini example to get familiar in more details with tenstorrent architecture.
In order to build this example, the user must compile the tt-metal stack first. This was done by using lima on MacOS. After tt-metal libraries are available, the simple example can be compiled and executed.

The risc-v is a small example that can be used to play and learn more about the risc-v isa. This example can be built in bazel and run with:
./third_party/whisper/whisper --isa rv32imaf --xlen 32 --gdb $(bazel info -c dbg bazel-bin)/risc-v/tensix_vector_kernel