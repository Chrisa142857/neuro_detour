


## Data: hcpa 

|    | backbone   | classifier   | data type        | acc              | f1               |
|---:|:-----------|:-------------|:-----------------|:-----------------|:-----------------|
|  0 | bnt        | gcn          | statfc aal       | 0.91116+-0.00639 | 0.91064+-0.00679 |
|  1 | bnt        | gcn          | statfc gordon    | 0.93979+-0.01648 | 0.93936+-0.01721 |
|  2 | bnt        | gcn          | dynfc aal        | 0.86681+-0.00584 | 0.86252+-0.00647 |
|  3 | bnt        | gcn          | dynfc gordon     | 0.92386+-0.00791 | 0.92334+-0.00892 |
|  4 | braingnn   | gcn          | dynfc gordon     | 0.75492+-0.09648 | 0.67576+-0.16355 |
|  5 | braingnn   | gcn          | statfc aal       | 0.73298+-0.12699 | 0.67655+-0.20136 |
|  6 | braingnn   | gcn          | statfc gordon    | 0.78305+-0.02389 | 0.77707+-0.02450 |
|  7 | braingnn   | gcn          | dynfc aal        | 0.77864+-0.05488 | 0.73315+-0.10367 |
|  8 | ndtNode    | gcn          | aal statfcBOLD   | 0.70068+-0.02193 | 0.62294+-0.05287 |
|  9 | ndtNode    | gcn          | gordon statfc    | 0.14250+-0.00106 | 0.03555+-0.00050 |
| 10 | none       | mlp          | statfc aal       | 0.58401+-0.00189 | 0.43151+-0.00340 |
| 11 | none       | mlp          | statfc gordon    | 0.58463+-0.00265 | 0.43139+-0.00320 |
| 12 | none       | gcn          | statfc gordon    | 0.92202+-0.00888 | 0.92193+-0.00916 |
| 13 | none       | mlp          | statfcBOLDaal    | 0.93418+-0.00578 | 0.93425+-0.00576 |
| 14 | none       | gcn          | statfcBOLDaal    | 0.92944+-0.00579 | 0.92979+-0.00603 |
| 15 | none       | gcn          | statfcBOLDgordon | 0.97427+-0.00508 | 0.97418+-0.00511 |
| 16 | none       | mlp          | statfcBOLDgordon | 0.97638+-0.00706 | 0.97633+-0.00697 |
| 17 | none       | gcn          | statscBOLDaal    | 0.92822+-0.00350 | 0.92837+-0.00375 |
| 18 | none       | gcn          | dynfc aal        | 0.85213+-0.00590 | 0.84847+-0.00777 |
| 19 | none       | mlp          | dynfcBOLDaal     | 0.83638+-0.01331 | 0.82862+-0.01677 |
| 20 | none       | gcn          | dynfcBOLDaal     | 0.84597+-0.00452 | 0.83945+-0.00365 |
| 21 | none       | gcn          | dynfcBOLDgordon  | 0.88294+-0.00278 | 0.88206+-0.00317 |
| 22 | none       | mlp          | dynfcBOLDgordon  | 0.89064+-0.00441 | 0.89004+-0.00314 |
| 23 | none       | gcn          | dynscBOLDaal     | 0.82663+-0.01102 | 0.82067+-0.01219 |
| 24 | none       | gcn          | dynfc gordon     | 0.88258+-0.00739 | 0.88194+-0.00751 |
| 25 | none       | mlp          | dynfc gordon     | 0.68770+-0.00430 | 0.56046+-0.00558 |
| 26 | none       | mlp          | dynfc aal        | 0.68709+-0.00382 | 0.56055+-0.00569 |
| 27 | none       | gcn          | statfc aal       | 0.90500+-0.00912 | 0.90485+-0.00938 |
| 28 | none       | gcn          | dynscBOLDgordon  | 0.88289+-0.00467 | 0.88150+-0.00445 |
| 29 | none       | gcn          | statscBOLDgordon | 0.97592+-0.00423 | 0.97596+-0.00416 |


## Data: adni 

|    | backbone   | classifier   | data type     | acc              | f1               |
|---:|:-----------|:-------------|:--------------|:-----------------|:-----------------|
|  0 | bnt        | gcn          | dynfc         | 0.83903+-0.06037 | 0.80331+-0.07912 |
|  1 | bnt        | gcn          | statfc        | 0.82741+-0.05889 | 0.79720+-0.06166 |
|  2 | braingnn   | gcn          | statfc        | 0.82815+-0.07899 | 0.76541+-0.10395 |
|  3 | ndtEdge    | gcn          | statfc        | 0.82074+-0.06857 | 0.75095+-0.08164 |
|  4 | ndtEdge    | gcn          | dynfc         | 0.83305+-0.05263 | 0.75789+-0.07409 |
|  5 | none       | mlp          | statfc aal    | 0.82815+-0.06466 | 0.75678+-0.08937 |
|  6 | none       | gcn          | statfcBOLDaal | 0.80667+-0.07260 | 0.76186+-0.08499 |
|  7 | none       | mlp          | statfcBOLDaal | 0.80667+-0.07260 | 0.74956+-0.09167 |
|  8 | none       | gcn          | statscBOLDaal | 0.81333+-0.07598 | 0.75608+-0.07678 |
|  9 | none       | gcn          | statfc aal    | 0.80074+-0.10523 | 0.74212+-0.09411 |
| 10 | none       | mlp          | dynfcBOLDaal  | 0.82934+-0.06353 | 0.76746+-0.07081 |
| 11 | none       | gcn          | dynfcBOLDaal  | 0.83533+-0.05393 | 0.77868+-0.05658 |
| 12 | none       | gcn          | dynscBOLDaal  | 0.80570+-0.04973 | 0.76675+-0.06839 |
| 13 | none       | gcn          | dynfc aal     | 0.83191+-0.05624 | 0.76736+-0.06470 |
| 14 | none       | mlp          | dynfc aal     | 0.83305+-0.05263 | 0.76635+-0.08668 |


## Data: oasis 

|    | backbone   | classifier   | data type   | acc              | f1               |
|---:|:-----------|:-------------|:------------|:-----------------|:-----------------|
|  0 | bnt        | gcn          | statfc      | 0.87060+-0.05246 | 0.85985+-0.05027 |
|  1 | bnt        | gcn          | dynfc       | 0.89953+-0.05062 | 0.88198+-0.05546 |
|  2 | braingnn   | gcn          | statfc      | 0.88563+-0.02620 | 0.84461+-0.05316 |
|  3 | braingnn   | gcn          | dynfc       | 0.89501+-0.02979 | 0.85494+-0.04065 |
|  4 | none       | gcn          | dynfcBOLD   | 0.88494+-0.03158 | 0.85858+-0.03969 |
|  5 | none       | gcn          | dynfc       | 0.91135+-0.03046 | 0.90457+-0.03171 |
|  6 | none       | mlp          | dynfc       | 0.91584+-0.02901 | 0.90825+-0.03796 |
|  7 | none       | gcn          | statfcBOLD  | 0.88272+-0.04873 | 0.85561+-0.05550 |
|  8 | none       | mlp          | statfcBOLD  | 0.88986+-0.03524 | 0.85268+-0.04820 |
|  9 | none       | mlp          | dynfcBOLD   | 0.89022+-0.03254 | 0.85020+-0.05028 |
| 10 | none       | mlp          | statfc      | 0.91597+-0.04441 | 0.90841+-0.05314 |
| 11 | none       | gcn          | statfc      | 0.89537+-0.02218 | 0.86805+-0.03885 |


## Data: ukb 

|    | backbone   | classifier   | data type        | acc              | f1               |
|---:|:-----------|:-------------|:-----------------|:-----------------|:-----------------|
|  0 | bnt        | gcn          | statfc aal       | 0.97976+-0.00319 | 0.97974+-0.00321 |
|  1 | braingnn   | gcn          | dynfc gordon     | 0.94118+-0.01506 | 0.94113+-0.01497 |
|  2 | braingnn   | gcn          | statfc aal       | 0.91917+-0.04329 | 0.91912+-0.04322 |
|  3 | braingnn   | gcn          | dynfc aal        | 0.84677+-0.11398 | 0.81929+-0.17316 |
|  4 | braingnn   | gcn          | statfc gordon    | 0.91759+-0.04232 | 0.91719+-0.04279 |
|  5 | none       | gcn          | statfc aal       | 0.97812+-0.00260 | 0.97811+-0.00259 |
|  6 | none       | mlp          | statfc gordon    | 0.55053+-0.01873 | 0.48211+-0.06286 |
|  7 | none       | gcn          | statfc gordon    | 0.98352+-0.00361 | 0.98350+-0.00361 |
|  8 | none       | gcn          | dynfc aal        | 0.94686+-0.00742 | 0.94669+-0.00739 |
|  9 | none       | gcn          | statfcBOLDaal    | 0.95663+-0.00754 | 0.95669+-0.00753 |
| 10 | none       | mlp          | statfcBOLDaal    | 0.93529+-0.00750 | 0.93538+-0.00750 |
| 11 | none       | gcn          | statfcBOLDgordon | 0.99305+-0.00328 | 0.99305+-0.00328 |
| 12 | none       | mlp          | statfcBOLDgordon | 0.99051+-0.00484 | 0.99051+-0.00485 |
| 13 | none       | gcn          | statscBOLDaal    | 0.92613+-0.00334 | 0.92637+-0.00334 |
| 14 | none       | mlp          | dynfc aal        | 0.62165+-0.02688 | 0.54222+-0.02438 |
| 15 | none       | mlp          | statfc aal       | 0.56252+-0.00852 | 0.41859+-0.01967 |
| 16 | none       | mlp          | dynfcBOLDaal     | 0.84356+-0.02103 | 0.84096+-0.02447 |
| 17 | none       | gcn          | dynfcBOLDaal     | 0.87469+-0.03356 | 0.87371+-0.03410 |
| 18 | none       | gcn          | dynfcBOLDgordon  | 0.93387+-0.00710 | 0.93357+-0.00705 |
| 19 | none       | mlp          | dynfcBOLDgordon  | 0.96420+-0.00598 | 0.96415+-0.00601 |
| 20 | none       | gcn          | dynscBOLDaal     | 0.86135+-0.00369 | 0.86044+-0.00419 |
| 21 | none       | gcn          | dynscBOLDgordon  | 0.96344+-0.00680 | 0.96345+-0.00678 |
| 22 | none       | gcn          | statscBOLDgordon | 0.99288+-0.00247 | 0.99289+-0.00247 |
| 23 | none       | gcn          | dynfc gordon     | 0.95894+-0.00343 | 0.95887+-0.00348 |
| 24 | none       | mlp          | dynfc gordon     | 0.62050+-0.02402 | 0.56111+-0.00957 |
