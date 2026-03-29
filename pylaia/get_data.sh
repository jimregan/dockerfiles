wget https://github.com/jimregan/tesseract-gle-uncial/releases/download/v0.1beta4_gen/seanchlo_generated.zip
unzip seanchlo_generated.zip
mkdir data/images
for set in train val test
do
	mkdir data/images/$set
	for i in $(cat data/$set.ids)
	do
		mv out/$i.jpg data/images/$set/
	done
done	
