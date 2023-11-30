# Evaluate a model on its own dev and test set (benchmark) and on others too (cross-lingual)
declare -a arr=("chu" "cop" "fro" "got" "grc" "hbo" "isl" "lat" "latm" "lzh" "ohu" "orv" "san")
for i in "${arr[@]}"
do
    !pip install https://huggingface.co/ljvmiranda921/xx_"$i"_sigtyp_trf/resolve/main/xx_"$i"_sigtyp_trf-any-py3-none-any.whl
    for j in "${arr[@]}"
    do
        weasel run evaluate-model . --vars.train_lang "$i" --vars.dev_lang "$j" --vars.test_lang "$j"
    done
done

