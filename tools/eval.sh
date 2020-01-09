name="_predicts.json"
for i in `seq 1 20`
do
  echo $i$name
  python3 tools/ai_format_kps_eval.py --ref jsons/val.json --submit /home/hsw/server/spm/jsons/$i$name
done
