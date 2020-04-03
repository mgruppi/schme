# Get data from SemEval2020

data=test_data_public
mkdir $data

# English
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
unzip -n semeval2020_ulscd_eng.zip
rm -rf $data/english
mv semeval2020_ulscd_eng $data/english
rm semeval2020_ulscd_eng.zip

# German
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
unzip -n semeval2020_ulscd_ger.zip
rm -rf $data/german
mv semeval2020_ulscd_ger $data/german
rm semeval2020_ulscd_ger.zip

# Latin
wget https://zenodo.org/record/3734089/files/semeval2020_ulscd_lat.zip
unzip -n semeval2020_ulscd_lat.zip
rm -rf $data/latin
mv semeval2020_ulscd_lat $data/latin
rm semeval2020_ulscd_lat.zip

# Swedish
wget https://zenodo.org/record/3730550/files/semeval2020_ulscd_swe.zip
unzip -n semeval2020_ulscd_swe.zip
rm -rf $data/swedish
mv semeval2020_ulscd_swe $data/swedish
rm semeval2020_ulscd_swe.zip
