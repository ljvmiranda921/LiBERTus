# Finetune a multi-task model for each language

weasel run convert-to-spacy --force
declare -a dev=("chu" "cop" "fro" "got" "grc" "hbo" "isl" "lat" "latm" "lzh" "ohu" "orv" "san")

# Common cases
declare -a arr=("chu" "fro" "got" "grc" "isl" "lat" "latm" "orv" "san")
for i in "${arr[@]}"
do
    weasel run finetune-trf-model . --vars.train_lang "$i" --vars.dev_lang "$i" 
    weasel run package-model . --vars.train_lang "$i"
    for j in "${dev[@]}"
    do
        weasel run evaluate-model-dev . --vars.train_lang "$i" --vars.dev_lang "$j"
        cp metrics/system_scores/"$i"/metrics-"$j"-dev.json /content/drive/MyDrive/Documents/Experiments/libertus/"$i"/
    done
done

# Special cases
## ohu needs to adjust the minimum frequency
spacy train configs/transformer.cfg --output-path training/ohu/ --paths.train corpus/train/ohu_train.spacy --paths.dev corpus/dev/ohu_valid.spacy --system.seed 42 --gpu-id 0 --components.parser.min_action_freq 1 --components.transformer.name ljvmiranda/LiBERTus-base
weasel run package-model . --vars.train_lang ohu
for j in "${dev[@]}"
do
    weasel run evaluate-model-dev . --vars.train_lang ohu --vars.dev_lang "$j"
    cp metrics/system_scores/ohu/metrics-"$j"-dev.json /content/drive/MyDrive/Documents/Experiments/libertus/ohu/
done

## cop needs special tokenizer
weasel run finetune-trf-model . --vars.train_lang cop --vars.dev_lang cop --vars.nlp_lang el --vars.config transformer.cfg
weasel run package-model . --vars.train_lang cop
for j in "${dev[@]}"
do
    weasel run evaluate-model-dev . --vars.train_lang cop --vars.dev_lang "$j"
    cp metrics/system_scores/cop/metrics-"$j"-dev.json /content/drive/MyDrive/Documents/Experiments/libertus/cop/
done

## lzh needs special tokenizer
weasel run finetune-trf-model . --vars.train_lang lzh --vars.dev_lang lzh --vars.nlp_lang zh --vars.config transformer_zh.cfg
weasel run package-model . --vars.train_lang lzh
for j in "${dev[@]}"
do
    weasel run evaluate-model-dev . --vars.train_lang lzh --vars.dev_lang "$j"
    cp metrics/system_scores/lzh/metrics-"$j"-dev.json /content/drive/MyDrive/Documents/Experiments/libertus/lzh/
done

## hbo might need some lexical info
weasel run finetune-trf-model . --vars.train_lang hbo --vars.dev_lang hbo --vars.nlp_lang he --vars.config transformer.cfg
weasel run package-model . --vars.train_lang hbo
for j in "${dev[@]}"
do
    weasel run evaluate-model-dev . --vars.train_lang hbo --vars.dev_lang "$j"
    cp metrics/system_scores/hbo/metrics-"$j"-dev.json /content/drive/MyDrive/Documents/Experiments/libertus/hbo/
done

## latm is weird because you need to upload it properly on HugginGface
weasel run finetune-trf-model . --vars.train_lang latm --vars.dev_lang latm
weasel run package-model . --vars.train_lang latm
cp packages/xx_latm_sigtyp_trf-0.1.0/dist/latm_sigtyp_trf-0.1.0-py3-none-any.whl /content/drive/MyDrive/