# Finetune a multi-task model for each language

weasel run convert-to-spacy --force

# Common cases
declare -a arr=("chu" "cop" "fro" "got" "grc" "hbo" "isl" "lat" "latm" "orv" "san")
for i in "${arr[@]}"
do
    weasel run finetune-trf-model . --vars.train_lang "$i" --vars.dev_lang "$i" 
    weasel run package-model . --vars.train_lang "$i"
done

# Special cases
# ohu needs to adjust the minimum frequency
spacy train configs/transformer.cfg --output-path training/ohu/ --paths.train corpus/train/ohu_train.spacy --paths.dev corpus/dev/ohu_valid.spacy --system.seed 42 --gpu-id 0 --components.parser.min_action_freq 1 --components.transformer.name ljvmiranda/LiBERTus-base
weasel run package-model . --vars.train_lang ohu
# lzh needs special tokenizer
weasel run finetune-trf-model . --vars.train_lang lzh --vars.dev_lang lzh --vars.nlp_lang zh --vars.config transformer.cfg
weasel run package-model . --vars.train_lang lzh
# hbo might need some lexical info
weasel run finetune-trf-model . --vars.train_lang hbo --vars.dev_lang hbo --vars.nlp_lang he --vars.config transformer.cfg
weasel run package-model . --vars.train_lang hbo
# latm is weird because you need to upload it properly on HugginGface
weasel run finetune-trf-model . --vars.train_lang latm --vars.dev_lang latm
weasel run package-model . --vars.train_lang latm
cp packages/xx_latm_sigtyp_trf-0.1.0/dist/latm_sigtyp_trf-0.1.0-py3-none-any.whl /content/drive/MyDrive/