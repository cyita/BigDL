set GGUF_PATH="D:\yina\yina-model-weights\ggml-model-llama3.2b-instruct-pure-q4_0.gguf"

@REM D:\yina\llama-cpp-bigdl\build\bin\Release\convert-gguf-to-npu.exe -m %GGUF_PATH% -o D:\yina\BigDL\llama3.2-3b-q40 --low-bit sym_int4

@REM D:\yina\BigDL\python\llm\example\NPU\HF-Transformers-AutoModels\LLM\llamacpp-convert\build\Release\convert-gguf-to-npu.exe -m %GGUF_PATH% -o D:\yina\BigDL\llama3.2-3b-q40 --low-bit sym_int4

build\Release\convert-gguf-to-npu.exe -m %GGUF_PATH% -o D:\yina\BigDL\python\llm\example\NPU\HF-Transformers-AutoModels\LLM\CPP_Examples\llama3.2-3b-q40 --low-bit sym_int4