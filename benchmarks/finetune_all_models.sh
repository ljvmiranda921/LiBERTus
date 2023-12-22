# Finetune a multi-task model for each language
declare -a arr=("chu" "cop" "fro" "got" "grc" "hbo" "isl" "lat" "latm" "lzh" "orv" "san")
weasel run convert-to-spacy --force
for i in "${arr[@]}"
do
    weasel run finetune-trf-model . --vars.train_lang "$i" --vars.dev_lang "$i" 
    weasel run package-model . --vars.train_lang "$i"
done

# Special cases
# ohu needs to adjust the minimum frequency
spacy train configs/transformer.cfg --output-path training/ohu/ --paths.train corpus/train/ohu_train.spacy --paths.dev corpus/dev/ohu_valid.spacy --system.seed 42 --gpu-id 0 --components.parser.min_action_freq 1 --components.transformer.name ljvmiranda/LiBERTus-base
weasel run package-model . --vars.train_lang ohu
# chinese needs special tokenizer
weasel run finetune-trf-model . --vars.train_lang lzh --vars.dev_lang lzh --vars.nlp_lang zh --vars.config transformer_zh.cfg
weasel run package-model . --vars.train_lang lzh
# hebrew needs all the help it can get