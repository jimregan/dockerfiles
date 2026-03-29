wget https://github.com/jimregan/tesseract-gle-uncial/releases/download/v0.1beta4_gen/seanchlo_generated.zip
unzip seanchlo_generated.zip
mkdir images
for set in train val test
do
	mkdir images/$set
	for i in $(cat data/$set.ids)
	do
		mv out/$i.jpg images/$set/
	done
done	
