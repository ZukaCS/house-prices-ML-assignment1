# house-prices-ML-assignment1
## კონკურსის მიმოხილვა

House Prices კონკურსის მიზანია სახლების ფასების პროგნოზირება სხვადასხვა მოცემული მახასიათებლების საფუძველზე. ამოცანა ფასდება (Root-Mean-Squared-Error)-ის ლოგარითმის მიხედვით predicted და რეალურ ფასებს შორის.


## მიდგომა
ეს ამოცანა სათაურის მიხედვით და ისედაც, ცხადია, რომ რეგრესიის ამოცანაა რადგან სახლის ფასები რიცხვების უწყვეტ ინტერვალზეა. შესაბამისად უნდა გამოვიყენოთ რეგრესიის მოდელები. ამისთვის თავიდან უნდა დავამუშაოთ შემოსული data. თავიდან ვყოფ data-ს 80/20 train და test ად. EDA- ფაზაში ვნახულობ ზოგადად data-ს თვისებებს, ასევე ვნახულობ თუ რამდენი სვეტია ისეთი რომელსაც ბევრი NaN მნიშვნელობა აქვს და შესაბამისად ვექცევი მონაცემებს (Cleaning & Preprocessing). შემდეგი ფაზაა Feature Engineering-ი სადაც ახალი ცვლადები შემომაქვს და რამდენიმე ძველ ცვლადებს გადავყრი. ასევე ამ ფაზაში ვიყენებ ისეთ მეთოდებს რითიც, კატეგორიულ ცვლადებს რიცხვითში გარდავქმნი, რადგან ეს საჭიროა, რომ რეგრესიის მოდელებმა იმუშაონ. შემდეგ მოდის Feature Selection -ფაზა სადაც რამდენიმე მეთოდი მაქვს რითიც მნიშვნელოვან ცვლადებს ამოვარჩევ. და საბოლოოდ მოდის ტრენინგის და ექსპერიმენტების ფაზა, სადაც ვტესტავ რამდენიმე რეგრესიის მოდელს, ვუკეთებ მათ თავიანთ pipeline-ებს და ვლოგავ mlflow -თი dagshub-ზე.

## რეპოზიტორიის სტრუქტურა

```
house-prices-ml/
├── house-prices-model-experiment.ipynb    ← EDA, data cleaning, feature engineering, ექსპერიმენტები
├── house-prices-model-inference.ipynb     ← საუკეთესო მოდელის ჩამოტვირთვა და submission
└── README.md
```
## ფაილების აღწერა

| ფაილი | აღწერა |
|-------|--------|
| `house-prices-model-experiment.ipynb` | ნოუთბუქი მონაცემების დამუშავებისთვის და ექსპერიმენტებისთვის |
| `house-prices-model-inference.ipynb` | საუკეთესო მოდელის ჩამოტვირთვა და kaggle-სთვის submission-ის შექმნა |
| `README.md` | პროექტის დოკუმენტაცია |


## მონაცემთა დამუშავება / გაწმენდა (Data preprocessing / cleaning )
### 1. NaN მნიშვნელობიანი სვეტების დამუშავება.
EDA -ს ფაზის შემდეგ, სადაც დატასეტი დავყავი ორად, train-ად და test-ად, train-ში მოვძებნე ისეთი სვეტები, რომლებსაც NaN მნიშვნელობები ჰქონდათ.

![NaN columns](images/nans.webp)

მიუხედავად იმისა, რომ ზოგიერთ დატასეტში NaN - ნიშნავს, რომ ეს თვისება უბრალოდ missing არის, ამ დატასეტში ზოგიერთი სვეტების NaN-ები ნიშნავს იმას, რომ სახლს უბრალოდ ეს თვისება არ აქვს. კეგლის data_description.txt შევამოწმე და ასეთი feature-ები იყო:

| სვეტი | მიზეზი |
|-------|--------|
| `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` | სახლს გარაჟი არ აქვს |
| `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` | სახლს სარდაფი არ აქვს |
| `PoolQC` | აუზი არ აქვს |
| `FireplaceQu` | ბუხარი არ აქვს |
| `Fence` | ღობე არ აქვს |
| `Alley` | გვერდითი შესასვლელი არ აქვს |
| `MiscFeature`, `MasVnrType` | შესაბამისი feature არ აქვს |

ამგვარი კატეგორიული სვეტები შევავსე უბრალოდ სტრინგი "None" -თი

| სვეტი | მიზეზი |
|-------|--------|
| `GarageYrBlt` | გარაჟი არ აქვს, ამიტომ აშენების წელი = 0 |

ასეთი რიცხვითი სვეტები კი უბრალოდ 0 ით შევავსე. 

### 2. ნამდვილად missing სვეტების შევსება.

დანარჩენი feature-ები კი, მაგალითად (`LotFrontage`, `MasVnrArea`, `Electrical`), რომელიც ზედა გრაფიკის მიხედვით შეიცავენ NaN-ებს, ნამდვილად დაკარგულია/არ არის აღრიცხული.

- SimpleImputer -ის გამოყენებით LotFrontage და MasVnrArea - რომლებიც, რიცხვითი სვეტებია, შევავსე მედიანით, რადგან ის უფრო მდგრადია outlier-ების მიმართ. 

- ხოლო კატეგორიული ცვლადის ('Electrical') NaN -ები - უბრალოდ შევავსე მოდით, ანუ ყველაზე ხშირი მნიშვნელობით. (აქაც SimpleImputer-ით).

## Feature Engineering

### 1. ახალი სვეტების შექმნა (`HouseFeatureAdderAndModifier`)

არსებული სვეტებიდან შევქმენი ახალი, უფრო ინფორმაციული feature-ები:

**ფართობის სვეტები:**
- `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF` — სახლის მთლიანი ფართობი
- `TotalPorchSF` = ყველა Porch-ის ფართობის ჯამი
- `BsmtFinTotal` = `BsmtFinSF1` + `BsmtFinSF2` — Finished სარდაფის ფართობი

**აბაზანის სვეტი:**
- `TotalBath` = `FullBath` + `BsmtFullBath` + `0.4×HalfBath` + `0.4×BsmtHalfBath` — სრული აბაზანა ალბათ უფრო მეტად ფასდება ამიტომ 0.4 ზე გავამრავლე პატარა აბაზანები.

**ასაკის სვეტები:**
- `HouseAge` = `YrSold` - `YearBuilt` — სახლის ასაკი გაყიდვის მომენტში
- `RemodelAge` = `YrSold` - `YearRemodAdd` — რემონტიდან გასული დრო

**ახალი სვეტები**
- `HasGarage` — აქვს თუ არა გარაჟი
- `HasBasement` — აქვს თუ არა სარდაფი
- `HasPool` — აქვს თუ არა აუზი
- `Has2ndFloor` — აქვს თუ არა მეორე სართული
- `WasRemodeled` — ჩაუტარდა თუ არა რემონტი

ამ ახალი სვეტების შექმნის შემდეგ, ის სვეტები რაც მათ შესაქმნელად გამოვიყენე, წავშალე რათა თავიდან ამერიდებინა collinear-ულობა ცვლადებს შორის. რეალურად ისინი აღარ იყო საჭირო.

შესაბამისად გადავყარე სვეტები: `1stFlrSF`, `2ndFlrSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `FullBath`, `HalfBath`, `BsmtFullBath`, `BsmtHalfBath`, `BsmtFinSF1`, `BsmtFinSF2`, `YearBuilt`, `YearRemodAdd`

            
---

### 2. Ordinal Encoding (`OrdinalEncoder`)

ზოგიერთ კატეგორიულ სვეტს აქვს **ბუნებრივი თანმიმდევრობა** — მაგალითად ხარისხი: Poor → Fair → Average → Good → Excellent. ამ სვეტებისთვის One-Hot Encoding არ არის შესაფერისი, რადგან ის კარგავს ამ თანმიმდევრობის ინფორმაციას.

ამ სვეტებისთვის გამოვიყენე Ordinal Encoding:

| სვეტები | მაპინგი |
|---------|--------|
| `GarageQual`, `GarageCond`, `PoolQC`, `ExterQual`, `ExterCond`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `BsmtQual`, `FireplaceQu` | `None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5` |
| `GarageFinish` | `None=0, Unf=1, RFn=2, Fin=3` |
| `PavedDrive` | `N=0, P=1, Y=2` |
| `Functional` | `Sal=1 ... Typ=8` |
| `LandSlope` | `Sev=1, Mod=2, Gtl=3` |

დარჩენილი კატეგორიული სვეტები დამუშავდა **One-Hot Encoding**-ით.

