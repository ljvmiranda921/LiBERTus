declare -a arr=("chu" "cop" "fro" "got" "grc" "hbo" "isl" "lat" "latm" "lzh" "ohu" "orv" "san")
for i in "${arr[@]}"
do
    weasel run finetune-model . --vars.train_lang "$i" --vars.dev_lang "$i" 
done