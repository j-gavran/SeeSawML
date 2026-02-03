# Exporting to ONNX

A model trained using SeesawML can be exported to the ONNX format for interoperability with other frameworks and tools. This is done using the `onnx_signal` or `onnx_fakes` commands after setting the checkpoint path `model_save_path` and checkpoint name `load_checkpoint` in the model configuration YAML file located in the `model_config/` directory.

!!! Warning "Not Yet Supported"
    ONNX export currently supports flat features only. Jagged features are not yet *fully* supported.
