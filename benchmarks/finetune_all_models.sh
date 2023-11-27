declare -a arr=("chu" "cop" "fro" "got" "grc" "hbo" "isl" "lat" "latm" "lzh" "ohu" "orv" "san")
weasel run convert-to-spacy --force
for i in "${arr[@]}"
do
    weasel run finetune-model . --vars.train_lang "$i" --vars.dev_lang "$i" 
    weasel run package-model . --vars.train_lang "$i"
    weasel push gcs
done