year='2021'

for month in {1..12}
do
    if (($month<10))
    then 
        python combine_for_icos.py ${year}0${month}
    else
        python combine_for_icos.py ${year}${month}
    fi
done
