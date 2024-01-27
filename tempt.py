from unimumo.audio.audiocraft_.models.loaders import load_mm_lm_model
from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes

model = load_mm_lm_model('facebook/musicgen-small', device='cpu')

descriptions = ['<music_prompt_start> Hi Hi Hi <music_prompt_end> <motion_prompt_start> <motion_prompt_end>',
                "<music_prompt_start> I love you <music_prompt_end> <motion_prompt_start> <motion_prompt_end>"]
attributes = [ConditioningAttributes(text={'description': description}) for description in descriptions]

attributes = model.cfg_dropout(attributes)
attributes = model.att_dropout(attributes)

tokenized = model.condition_provider.tokenize(attributes, device='cpu')

print('here')




