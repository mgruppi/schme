
#!bin/bash
declare -A news_groups
news_groups=( ["fasttext_300_Infowars-The Gateway Pundit-Veterans Today-Newswars-Prison Planet"]="news_group1"
        ["fasttext_300_National Review-Fox News-Real Clear Politics-Washington Examiner-Investors Business Daily"]="news_group4"
        ["fasttext_300_Reuters-CNN-CBS News-ABC News-The Hill"]="news_group3"
        ["fasttext_300_RT-Sputnik-Tass-Russia-Insider-Fort Russ"]="news_group6"
        ["fasttext_300_Talking Points Memo-Raw Story-Daily Kos-Crooks and Liars-Media Matters for America"]="news_group2"
        ["fasttext_300_The Daily Caller-Breitbart-RedState-Hot Air-TheBlaze"]="news_group5"
        ["fasttext_300_The Guardian UK-The Independent-Evening Standard-BBC UK-The Daily Mirror-The Sun-Birmingham Mail"]="news_group7")

parent_dir="raw"
for i in $parent_dir/*; do
  for j in "$i"/*.txt ;
    do mv -- "$j" "${j%.txt}.vec";
  done  ;
done


for i in $parent_dir/*; do
  for j in "$i"/*.vec ; do
    echo $(basename "$j") "${news_groups[$(basename "${j%.vec}")]}"
    mv -- "$j" $(dirname "$j")/"${news_groups["$(basename "${j%.vec}")"]}.vec"
  done
done
