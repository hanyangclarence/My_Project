from unimumo.audio.audiocraft_.models.loaders import load_mm_lm_model

model = load_mm_lm_model("facebook/musicgen-small", device='cpu')

result = model.generate(max_gen_len=30)
print(result)

