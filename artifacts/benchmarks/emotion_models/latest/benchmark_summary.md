# Emotion Model Benchmark

| Model | Type | Params | Public Test | Webcam Test | Latency (ms) | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| convnextv2_pico | pretrained backbone | 8556358 | 0.8930 | 0.9927 | 11.98 | Strongest class: sad (F1 1.000); Weakest class: fear (F1 0.979) |
| resnet18 | pretrained backbone | 11179590 | 0.8289 | 0.9927 | 6.49 | Strongest class: anger (F1 1.000); Weakest class: happy (F1 0.968) |
| cnn | custom baseline CNN | 422086 | 0.2299 | 0.5092 | 7.42 | Strongest class: neutral (F1 0.675); Weakest class: anger (F1 0.000) |
