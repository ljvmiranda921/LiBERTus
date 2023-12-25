# Create control variable dev set
python -m scripts.convert_to_pretrain assets/dev --output-path corpus/control_variable_dev.txt --seed 42 --shuffle --verbose
# Upsampling
python -m scripts.convert_to_pretrain assets/train --output-path corpus/up_sampling_train.txt --sampling-strategy upsample --seed 42 --shuffle --verbose --num-choices 1000
weasel run experiment-sampling . --vars.pretraining_corpus corpus/up_sampling_train.txt --vars.evaluation_corpus corpus/control_variable_dev.txt --vars.name up_sampling --vars.max_steps 20000
# Averaging
python -m scripts.convert_to_pretrain assets/train --output-path corpus/avg_sampling_train.txt --sampling-strategy average --seed 42 --shuffle --verbose --num-choices 1000
weasel run experiment-sampling . --vars.pretraining_corpus corpus/avg_sampling_train.txt --vars.evaluation_corpus corpus/control_variable_dev.txt --vars.name avg_sampling --vars.max_steps 20000 --vars.seed 1010