


## Data: mlp 

|    | backbone   | classifier   | data type   | acc              | f1               |
|---:|:-----------|:-------------|:------------|:-----------------|:-----------------|
|  0 | none       | adni-oasis   | statfcBOLD  | 0.86524+-0.04456 | 0.79602+-0.06557 |
|  1 | none       | adni-oasis   | dynfcBOLD   | 0.84252+-0.03775 | 0.80360+-0.05719 |
|  2 | none       | oasis-adni   | dynfcBOLD   | 0.80940+-0.06589 | 0.70630+-0.11197 |
|  3 | none       | oasis-adni   | statfcBOLD  | 0.80519+-0.05663 | 0.73151+-0.09577 |


## Data: gcn 

|    | backbone    | classifier   | data type   | acc              | f1               |
|---:|:------------|:-------------|:------------|:-----------------|:-----------------|
|  0 | neurodetour | adni-oasis   | dynfcBOLD   | 0.79650+-0.06345 | 0.81344+-0.04857 |
|  1 | none        | adni-oasis   | statfcBOLD  | 0.88236+-0.03314 | 0.82276+-0.06391 |
|  2 | none        | adni-oasis   | dynfcBOLD   | 0.86643+-0.04910 | 0.81011+-0.06141 |
|  3 | graphormer  | adni-oasis   | dynfcBOLD   | 0.83546+-0.06896 | 0.81295+-0.04834 |
|  4 | nagphormer  | adni-oasis   | statfcBOLD  | 0.69629+-0.10992 | 0.81069+-0.04456 |
|  5 | neurodetour | adni-oasis   | statfcBOLD  | 0.80034+-0.08500 | 0.80169+-0.05434 |
|  6 | graphormer  | adni-oasis   | statfcBOLD  | 0.79692+-0.07706 | 0.80359+-0.07494 |
|  7 | nagphormer  | adni-oasis   | dynfcBOLD   | 0.78087+-0.07083 | 0.81204+-0.05114 |
|  8 | bolt        | hcpa-ukb     | dynfcBOLD   | 0.89636+-0.00615 | 0.89712+-0.00710 |
|  9 | nagphormer  | hcpa-ukb     | statfcBOLD  | 0.62125+-0.06573 | 0.74492+-0.04008 |
| 10 | bolt        | hcpa-ukb     | statfcBOLD  | 0.86291+-0.03434 | 0.86514+-0.03386 |
| 11 | braingnn    | hcpa-ukb     | dynfcBOLD   | 0.67600+-0.02723 | 0.64733+-0.05787 |
| 12 | bnt         | hcpa-ukb     | statfcBOLD  | 0.76682+-0.08549 | 0.82877+-0.04502 |
| 13 | neurodetour | hcpa-ukb     | statfcBOLD  | 0.90607+-0.02702 | 0.91291+-0.02097 |
| 14 | neurodetour | hcpa-ukb     | statfcBOLD  | 0.66135+-0.14276 | 0.76168+-0.04656 |
| 15 | neurodetour | hcpa-ukb     | statfcBOLD  | 0.64588+-0.03229 | 0.78111+-0.01398 |
| 16 | bnt         | hcpa-ukb     | statfcBOLD  | 0.92301+-0.03929 | 0.92923+-0.02825 |
| 17 | bnt         | hcpa-ukb     | dynfcBOLD   | 0.82948+-0.10066 | 0.78714+-0.20357 |
| 18 | neurodetour | hcpa-ukb     | dynfcBOLD   | 0.71214+-0.01374 | 0.72079+-0.02147 |
| 19 | nagphormer  | hcpa-ukb     | statfcBOLD  | 0.91923+-0.00908 | 0.92042+-0.00790 |
| 20 | graphormer  | hcpa-ukb     | dynfcBOLD   | 0.38550+-0.06149 | 0.50967+-0.04008 |
| 21 | nagphormer  | hcpa-ukb     | dynfcBOLD   | 0.71374+-0.01174 | 0.70174+-0.01315 |
| 22 | braingnn    | hcpa-ukb     | statfcBOLD  | 0.75689+-0.06640 | 0.78586+-0.04322 |
| 23 | graphormer  | hcpa-ukb     | statfcBOLD  | 0.44062+-0.01101 | 0.39091+-0.28145 |
| 24 | graphormer  | oasis-adni   | dynfcBOLD   | 0.77236+-0.07337 | 0.72576+-0.08941 |
| 25 | graphormer  | oasis-adni   | statfcBOLD  | 0.77630+-0.02888 | 0.69920+-0.12324 |
| 26 | neurodetour | oasis-adni   | dynfcBOLD   | 0.81567+-0.07239 | 0.69316+-0.09157 |
| 27 | nagphormer  | oasis-adni   | dynfcBOLD   | 0.78091+-0.07243 | 0.70233+-0.08603 |
| 28 | none        | oasis-adni   | statfcBOLD  | 0.79111+-0.05169 | 0.67371+-0.11036 |
| 29 | nagphormer  | oasis-adni   | statfcBOLD  | 0.73111+-0.05900 | 0.70586+-0.08275 |
| 30 | none        | oasis-adni   | dynfcBOLD   | 0.81197+-0.05709 | 0.71601+-0.09172 |
| 31 | graphormer  | ukb-hcpa     | dynfcBOLD   | 0.51669+-0.16678 | 0.64364+-0.07898 |
| 32 | nagphormer  | ukb-hcpa     | statfcBOLD  | 0.63747+-0.01127 | 0.89772+-0.00094 |
| 33 | bolt        | ukb-hcpa     | statfcBOLD  | 0.91283+-0.00747 | 0.91550+-0.00772 |
| 34 | braingnn    | ukb-hcpa     | dynfcBOLD   | 0.68784+-0.01474 | 0.63613+-0.03837 |
| 35 | nagphormer  | ukb-hcpa     | statfcBOLD  | 0.93029+-0.01088 | 0.93226+-0.01076 |
| 36 | bnt         | ukb-hcpa     | statfcBOLD  | 0.86078+-0.03176 | 0.87040+-0.02899 |
| 37 | neurodetour | ukb-hcpa     | statfcBOLD  | 0.88726+-0.04995 | 0.90615+-0.03650 |
| 38 | neurodetour | ukb-hcpa     | dynfcBOLD   | 0.75905+-0.02472 | 0.75617+-0.02981 |
| 39 | graphormer  | ukb-hcpa     | statfcBOLD  | 0.51466+-0.08892 | 0.57780+-0.14501 |
| 40 | bnt         | ukb-hcpa     | dynfcBOLD   | 0.79109+-0.06155 | 0.74557+-0.15500 |
| 41 | braingnn    | ukb-hcpa     | statfcBOLD  | 0.81792+-0.03667 | 0.82497+-0.03273 |
| 42 | nagphormer  | ukb-hcpa     | dynfcBOLD   | 0.74548+-0.00647 | 0.73441+-0.00701 |
| 43 | neurodetour | ukb-hcpa     | statfcBOLD  | 0.65993+-0.03512 | 0.89976+-0.00271 |


## Data: ukb 

|    | backbone                | classifier   | data type         | acc              | f1               |
|---:|:------------------------|:-------------|:------------------|:-----------------|:-----------------|
|  0 | neurodetour4H2L         | gcn          | statfcFC gordon   | 0.99134+-0.00038 | 0.99133+-0.00037 |
|  1 | neurodetour4H1L         | gcn          | statfcFC gordon   | 0.99118+-0.00218 | 0.99117+-0.00219 |
|  2 | neurodetour3H2L         | gcn          | statfcFC gordon   | 0.99202+-0.00230 | 0.99201+-0.00230 |
|  3 | neurodetour3H1L         | gcn          | statfcFC gordon   | 0.99151+-0.00102 | 0.99151+-0.00102 |
|  4 | neurodetour3H1L         | gcn          | statfcFC gordon   | 0.99168+-0.00186 | 0.99167+-0.00187 |
|  5 | neurodetour6H2L         | gcn          | statfcFC gordon   | 0.99134+-0.00126 | 0.99134+-0.00126 |
|  6 | neurodetour6H1L         | gcn          | statfcFC gordon   | 0.99152+-0.00227 | 0.99152+-0.00227 |
|  7 | neurodetour5H2L         | gcn          | statfcFC gordon   | 0.99203+-0.00171 | 0.99202+-0.00171 |
|  8 | neurodetour5H1L         | gcn          | statfcFC gordon   | 0.99134+-0.00192 | 0.99134+-0.00192 |
|  9 | neurodetour1H2L         | gcn          | statfcFC gordon   | 0.99185+-0.00211 | 0.99185+-0.00211 |
| 10 | neurodetour1H1L         | gcn          | statfcFC gordon   | 0.99202+-0.00211 | 0.99202+-0.00211 |
| 11 | neurodetour1H1L         | gcn          | statfcFC gordon   | 0.99169+-0.00199 | 0.99168+-0.00199 |
| 12 | neurodetour2H1L         | gcn          | statfcBOLD gordon | 0.99594+-0.00208 | 0.99594+-0.00208 |
| 13 | neurodetour2H1L         | gcn          | dynfcFC aal       | 0.97146+-0.00365 | 0.97143+-0.00368 |
| 14 | neurodetour2H1L         | gcn          | dynfcBOLD aal     | 0.89068+-0.01207 | 0.88886+-0.01169 |
| 15 | neurodetour2H1L         | gcn          | dynfcBOLD gordon  | 0.94121+-0.00748 | 0.94105+-0.00752 |
| 16 | neurodetour2H1L         | gcn          | statfcBOLD aal    | 0.96333+-0.00678 | 0.96337+-0.00676 |
| 17 | neurodetour2H1L         | gcn          | dynfcFC gordon    | 0.97769+-0.00210 | 0.97765+-0.00210 |
| 18 | neurodetour2H1L         | gcn          | statfcFC aal      | 0.99014+-0.00202 | 0.99014+-0.00203 |
| 19 | neurodetour2H1L         | gcn          | statfcFC gordon   | 0.99220+-0.00238 | 0.99219+-0.00239 |
| 20 | neurodetour7H1L         | gcn          | statfcFC gordon   | 0.99219+-0.00112 | 0.99218+-0.00112 |
| 21 | none                    | gcn          | statfc aal        | 0.97812+-0.00260 | 0.97811+-0.00259 |
| 22 | none                    | gcn          | dynfc aal         | 0.94686+-0.00742 | 0.94669+-0.00739 |
| 23 | transformer             | gcn          | statfcBOLD aal    | 0.93687+-0.04178 | 0.93690+-0.04183 |
| 24 | none                    | gcn          | dynfcFC gordon    | 0.97537+-0.00241 | 0.97535+-0.00243 |
| 25 | none                    | gcn          | statfcFC aal      | 0.98869+-0.00326 | 0.98868+-0.00326 |
| 26 | none                    | gcn          | dynfcFC aal       | 0.97269+-0.00288 | 0.97260+-0.00293 |
| 27 | none                    | gcn          | dynfcFC aal       | 0.96988+-0.00389 | 0.96977+-0.00398 |
| 28 | none                    | gcn          | dynfcFC gordon    | 0.97670+-0.00265 | 0.97664+-0.00268 |
| 29 | transformer             | gcn          | statfcBOLD gordon | 0.99593+-0.00149 | 0.99593+-0.00149 |
| 30 | transformer             | gcn          | dynfcBOLD aal     | 0.88795+-0.02465 | 0.88765+-0.02507 |
| 31 | transformer             | gcn          | dynfcBOLD gordon  | 0.94160+-0.00614 | 0.94150+-0.00605 |
| 32 | transformer             | gcn          | statfcFC aal      | 0.99161+-0.00150 | 0.99160+-0.00150 |
| 33 | transformer             | gcn          | statfcBOLD gordon | 0.99627+-0.00220 | 0.99627+-0.00220 |
| 34 | none                    | gcn          | statfcBOLD aal    | 0.95663+-0.00754 | 0.95669+-0.00753 |
| 35 | neurodetourSingleSC2H1L | gcn          | statfcFC gordon   | 0.99100+-0.00132 | 0.99099+-0.00132 |
| 36 | neurodetourSingleFC2H1L | gcn          | statfcFC aal      | 0.99034+-0.00165 | 0.99033+-0.00165 |
| 37 | neurodetourSingleFC2H1L | gcn          | statfcFC gordon   | 0.99253+-0.00183 | 0.99253+-0.00183 |
| 38 | neurodetour8H2L         | gcn          | statfcFC gordon   | 0.99354+-0.00206 | 0.99354+-0.00206 |
| 39 | neurodetour8H1L         | gcn          | statfcFC gordon   | 0.99185+-0.00284 | 0.99185+-0.00284 |
| 40 | neurodetour7H2L         | gcn          | statfcFC gordon   | 0.99271+-0.00172 | 0.99270+-0.00173 |
| 41 | neurodetourSingleSC2H1L | gcn          | statfcFC aal      | 0.99070+-0.00343 | 0.99069+-0.00343 |
| 42 | none                    | gcn          | statfcBOLD gordon | 0.99305+-0.00328 | 0.99305+-0.00328 |
| 43 | none                    | gcn          | statfc gordon     | 0.98352+-0.00361 | 0.98350+-0.00361 |
| 44 | none                    | gcn          | dynfc gordon      | 0.95894+-0.00343 | 0.95887+-0.00348 |
| 45 | none                    | gcn          | statfcFC aal      | 0.98959+-0.00382 | 0.98959+-0.00382 |
| 46 | none                    | gcn          | dynfcBOLD aal     | 0.87469+-0.03356 | 0.87371+-0.03410 |
| 47 | none                    | gcn          | statfcFC gordon   | 0.98999+-0.00221 | 0.98999+-0.00221 |
| 48 | none                    | gcn          | dynfcBOLD gordon  | 0.93387+-0.00710 | 0.93357+-0.00705 |
| 49 | none                    | gcn          | statfcFC gordon   | 0.99101+-0.00193 | 0.99100+-0.00193 |
| 50 | braingnn                | gcn          | statfc gordon     | 0.91759+-0.04232 | 0.91719+-0.04279 |
| 51 | braingnn                | gcn          | statfcBOLD aal    | 0.92341+-0.00590 | 0.92366+-0.00587 |
| 52 | braingnn                | gcn          | dynfcFC aal       | 0.95528+-0.01703 | 0.95505+-0.01744 |
| 53 | bolt                    | gcn          | statfcFC gordon   | 0.99132+-0.00335 | 0.99131+-0.00335 |
| 54 | bolt                    | gcn          | statfcBOLD aal    | 0.98322+-0.00261 | 0.98322+-0.00262 |
| 55 | braingnn                | gcn          | dynfcBOLD aal     | 0.86045+-0.02481 | 0.85980+-0.02605 |
| 56 | braingnn                | gcn          | dynfc aal         | 0.84677+-0.11398 | 0.81929+-0.17316 |
| 57 | braingnn                | gcn          | statfcFC aal      | 0.98395+-0.00381 | 0.98393+-0.00381 |
| 58 | braingnn                | gcn          | dynfcBOLD gordon  | 0.86115+-0.04042 | 0.86087+-0.04180 |
| 59 | braingnn                | gcn          | dynfc gordon      | 0.94118+-0.01506 | 0.94113+-0.01497 |
| 60 | bolt                    | gcn          | dynfcFC aal       | 0.97213+-0.00364 | 0.97206+-0.00365 |
| 61 | bolt                    | gcn          | statfcFC aal      | 0.99016+-0.00250 | 0.99015+-0.00251 |
| 62 | bolt                    | gcn          | dynfcBOLD gordon  | 0.98222+-0.00311 | 0.98217+-0.00314 |
| 63 | bnt                     | gcn          | statfcFC gordon   | 0.98708+-0.00345 | 0.98708+-0.00345 |
| 64 | bnt                     | gcn          | dynfcBOLD aal     | 0.95285+-0.00743 | 0.95278+-0.00751 |
| 65 | bnt                     | gcn          | statfcBOLD gordon | 0.98643+-0.00184 | 0.98643+-0.00184 |
| 66 | bnt                     | gcn          | statfcFC aal      | 0.99089+-0.00217 | 0.99088+-0.00218 |
| 67 | bnt                     | gcn          | dynfcBOLD gordon  | 0.95981+-0.00438 | 0.95974+-0.00431 |
| 68 | bnt                     | gcn          | statfc aal        | 0.97976+-0.00319 | 0.97974+-0.00321 |
| 69 | bnt                     | gcn          | dynfcFC aal       | 0.97691+-0.00359 | 0.97684+-0.00363 |
| 70 | bolt                    | gcn          | dynfcBOLD aal     | 0.96917+-0.00430 | 0.96917+-0.00432 |
| 71 | bolt                    | gcn          | statfcBOLD gordon | 0.99287+-0.00257 | 0.99287+-0.00258 |
| 72 | bnt                     | gcn          | statfcBOLD aal    | 0.98122+-0.00205 | 0.98121+-0.00205 |
| 73 | bnt                     | gcn          | dynfcFC gordon    | 0.97155+-0.00492 | 0.97152+-0.00492 |
| 74 | braingnn                | gcn          | statfc aal        | 0.91917+-0.04329 | 0.91912+-0.04322 |
| 75 | braingnn                | gcn          | statfcBOLD gordon | 0.90327+-0.02722 | 0.90349+-0.02696 |
| 76 | nagphormer              | gcn          | statfcBOLD aal    | 0.95206+-0.03037 | 0.95204+-0.03046 |
| 77 | nagphormer              | gcn          | dynfcFC gordon    | 0.96832+-0.00357 | 0.96822+-0.00355 |
| 78 | nagphormer              | gcn          | statfcBOLD gordon | 0.99220+-0.00360 | 0.99220+-0.00359 |
| 79 | nagphormer              | gcn          | statfcFC gordon   | 0.98793+-0.00351 | 0.98792+-0.00352 |
| 80 | nagphormer              | gcn          | dynfcFC aal       | 0.96665+-0.00533 | 0.96660+-0.00541 |
| 81 | nagphormer              | gcn          | dynfcBOLD gordon  | 0.92902+-0.00691 | 0.92878+-0.00681 |
| 82 | nagphormer              | gcn          | statfcFC aal      | 0.98320+-0.00454 | 0.98319+-0.00454 |
| 83 | nagphormer              | gcn          | dynfcBOLD aal     | 0.84659+-0.01716 | 0.84155+-0.02065 |
| 84 | graphormer              | gcn          | statfcBOLD gordon | 0.86822+-0.12424 | 0.86545+-0.13026 |
| 85 | graphormer              | gcn          | dynfcFC aal       | 0.88922+-0.04381 | 0.88591+-0.04816 |
| 86 | graphormer              | gcn          | dynfcBOLD aal     | 0.71230+-0.09234 | 0.70837+-0.08277 |
| 87 | graphormer              | gcn          | statfcFC gordon   | 0.92759+-0.10045 | 0.92667+-0.10249 |
| 88 | graphormer              | gcn          | statfcBOLD aal    | 0.58586+-0.13626 | 0.55944+-0.17294 |
| 89 | graphormer              | gcn          | statfcFC aal      | 0.94606+-0.03812 | 0.94612+-0.03792 |
| 90 | graphormer              | gcn          | dynfcBOLD gordon  | 0.55560+-0.21083 | 0.50121+-0.27583 |
| 91 | graphormer              | gcn          | dynfcFC gordon    | 0.81983+-0.09830 | 0.80190+-0.11346 |
| 92 | none                    | mlp          | statfc aal        | 0.56252+-0.00852 | 0.41859+-0.01967 |
| 93 | none                    | mlp          | statfcBOLD aal    | 0.93529+-0.00750 | 0.93538+-0.00750 |
| 94 | none                    | mlp          | statfc gordon     | 0.55053+-0.01873 | 0.48211+-0.06286 |
| 95 | none                    | mlp          | dynfcBOLD gordon  | 0.96420+-0.00598 | 0.96415+-0.00601 |
| 96 | none                    | mlp          | dynfcBOLD aal     | 0.84356+-0.02103 | 0.84096+-0.02447 |
| 97 | none                    | mlp          | dynfc gordon      | 0.62050+-0.02402 | 0.56111+-0.00957 |
| 98 | none                    | mlp          | statfcBOLD gordon | 0.99051+-0.00484 | 0.99051+-0.00485 |
| 99 | none                    | mlp          | dynfc aal         | 0.62165+-0.02688 | 0.54222+-0.02438 |


## Data: oasis 

|    | backbone                | classifier   | data type   | acc              | f1               |
|---:|:------------------------|:-------------|:------------|:-----------------|:-----------------|
|  0 | neurodetour4H2L         | gcn          | statfcFC    | 0.88519+-0.04572 | 0.85341+-0.04788 |
|  1 | neurodetour4H1L         | gcn          | statfcFC    | 0.89515+-0.03994 | 0.85243+-0.06340 |
|  2 | neurodetour4H1L         | gcn          | statfcFC    | 0.89484+-0.03823 | 0.87164+-0.04587 |
|  3 | neurodetour3H2L         | gcn          | statfcFC    | 0.89041+-0.04500 | 0.86278+-0.04632 |
|  4 | neurodetour5H1L         | gcn          | statfcFC    | 0.88228+-0.05773 | 0.86653+-0.05362 |
|  5 | neurodetour3H1L         | gcn          | statfcFC    | 0.88996+-0.03113 | 0.85069+-0.03766 |
|  6 | neurodetour7H1L         | gcn          | statfcFC    | 0.89320+-0.04252 | 0.85881+-0.04611 |
|  7 | neurodetour6H2L         | gcn          | statfcFC    | 0.88301+-0.03819 | 0.86321+-0.03331 |
|  8 | neurodetour6H1L         | gcn          | statfcFC    | 0.88791+-0.02912 | 0.85947+-0.04092 |
|  9 | neurodetour5H1L         | gcn          | statfcFC    | 0.88584+-0.04776 | 0.85262+-0.04213 |
| 10 | neurodetour5H2L         | gcn          | statfcFC    | 0.88281+-0.04101 | 0.85543+-0.05412 |
| 11 | neurodetour1H1L         | gcn          | statfcFC    | 0.88216+-0.04073 | 0.85052+-0.03784 |
| 12 | neurodetour1H2L         | gcn          | statfcFC    | 0.88575+-0.04093 | 0.86335+-0.03891 |
| 13 | neurodetour1H1L         | gcn          | statfcFC    | 0.89787+-0.04460 | 0.86462+-0.05750 |
| 14 | neurodetour             | gcn          | statfc      | 0.88998+-0.03977 | 0.86082+-0.04522 |
| 15 | neurodetour             | gcn          | statfcFC    | 0.89005+-0.03205 | 0.86301+-0.03150 |
| 16 | neurodetour             | gcn          | dynfc       | 0.90877+-0.04071 | 0.89703+-0.04461 |
| 17 | neurodetour             | gcn          | dynfcBOLD   | 0.87121+-0.04631 | 0.84976+-0.04278 |
| 18 | neurodetour2H1L         | gcn          | dynfcFC     | 0.89494+-0.03328 | 0.86100+-0.04213 |
| 19 | neurodetour2H1L         | gcn          | dynfcBOLD   | 0.88125+-0.04411 | 0.85006+-0.04479 |
| 20 | neurodetour2H1L         | gcn          | statfcBOLD  | 0.88519+-0.04085 | 0.85955+-0.03440 |
| 21 | neurodetour2H1L         | gcn          | statfcFC    | 0.90013+-0.03419 | 0.86373+-0.05031 |
| 22 | neurodetour7H2L         | gcn          | statfcFC    | 0.88529+-0.04157 | 0.86443+-0.04463 |
| 23 | transformer             | gcn          | dynfcFC     | 0.89032+-0.03653 | 0.86810+-0.04485 |
| 24 | none                    | gcn          | statfc      | 0.89537+-0.02218 | 0.86805+-0.03885 |
| 25 | none                    | gcn          | statfcFC    | 0.88801+-0.02879 | 0.84747+-0.05561 |
| 26 | none                    | gcn          | dynfc       | 0.91135+-0.03046 | 0.90457+-0.03171 |
| 27 | transformer             | gcn          | dynfcBOLD   | 0.88071+-0.04459 | 0.85220+-0.04345 |
| 28 | transformer             | gcn          | statfc      | 0.90241+-0.03056 | 0.88760+-0.03182 |
| 29 | transformer             | gcn          | statfcFC    | 0.88517+-0.03483 | 0.86190+-0.03809 |
| 30 | transformer             | gcn          | statfcBOLD  | 0.89255+-0.02624 | 0.85877+-0.04890 |
| 31 | neurodetourSingleSC2H1L | gcn          | statfcFC    | 0.89051+-0.03987 | 0.86109+-0.04322 |
| 32 | neurodetourSingleFC2H1L | gcn          | statfcFC    | 0.89308+-0.04365 | 0.86583+-0.05867 |
| 33 | neurodetour8H2L         | gcn          | statfcFC    | 0.89547+-0.03503 | 0.85751+-0.05138 |
| 34 | neurodetour8H1L         | gcn          | statfcFC    | 0.88539+-0.02792 | 0.85868+-0.03992 |
| 35 | neurodetour8H1L         | gcn          | statfcFC    | 0.87728+-0.03875 | 0.84191+-0.05142 |
| 36 | none                    | gcn          | statfcBOLD  | 0.88272+-0.04873 | 0.85561+-0.05550 |
| 37 | none                    | gcn          | dynfcFC     | 0.88303+-0.03541 | 0.85713+-0.04096 |
| 38 | none                    | gcn          | dynfcBOLD   | 0.88494+-0.03158 | 0.85858+-0.03969 |
| 39 | braingnn                | gcn          | statfcFC    | 0.89289+-0.04746 | 0.86073+-0.05709 |
| 40 | bolt                    | gcn          | dynfcBOLD   | 0.88504+-0.03352 | 0.84702+-0.04395 |
| 41 | bolt                    | gcn          | statfcFC    | 0.88303+-0.03766 | 0.85543+-0.05299 |
| 42 | bolt                    | gcn          | dynfcFC     | 0.88968+-0.03035 | 0.85487+-0.03854 |
| 43 | braingnn                | gcn          | dynfcFC     | 0.89649+-0.03309 | 0.85118+-0.04905 |
| 44 | braingnn                | gcn          | dynfc       | 0.89501+-0.02979 | 0.85494+-0.04065 |
| 45 | braingnn                | gcn          | statfc      | 0.88563+-0.02620 | 0.84461+-0.05316 |
| 46 | braingnn                | gcn          | dynfcBOLD   | 0.89271+-0.03361 | 0.84503+-0.05005 |
| 47 | neurodetour             | gcn          | dynfcFC     | 0.88566+-0.03808 | 0.86255+-0.04482 |
| 48 | bnt                     | gcn          | dynfcBOLD   | 0.89568+-0.03018 | 0.85668+-0.04044 |
| 49 | bnt                     | gcn          | statfcFC    | 0.89017+-0.03480 | 0.86071+-0.03186 |
| 50 | bnt                     | gcn          | statfc      | 0.87060+-0.05246 | 0.85985+-0.05027 |
| 51 | bolt                    | gcn          | statfcBOLD  | 0.87543+-0.04624 | 0.84907+-0.04763 |
| 52 | bnt                     | gcn          | statfcBOLD  | 0.88748+-0.04359 | 0.85320+-0.04853 |
| 53 | bnt                     | gcn          | dynfcFC     | 0.89982+-0.02748 | 0.86728+-0.03570 |
| 54 | bnt                     | gcn          | dynfc       | 0.89953+-0.05062 | 0.88198+-0.05546 |
| 55 | nagphormer              | gcn          | statfc      | 0.89017+-0.03480 | 0.84261+-0.05227 |
| 56 | nagphormer              | gcn          | dynfc       | 0.90219+-0.03497 | 0.87802+-0.03297 |
| 57 | nagphormer              | gcn          | statfcFC    | 0.89527+-0.03327 | 0.85613+-0.04794 |
| 58 | graphormer              | gcn          | dynfcBOLD   | 0.88977+-0.03262 | 0.84397+-0.04771 |
| 59 | nagphormer              | gcn          | statfcBOLD  | 0.89017+-0.03480 | 0.83874+-0.05023 |
| 60 | nagphormer              | gcn          | dynfcBOLD   | 0.89208+-0.03441 | 0.84475+-0.04577 |
| 61 | neurodetour             | gcn          | statfcBOLD  | 0.89991+-0.02089 | 0.87024+-0.03772 |
| 62 | nagphormer              | gcn          | dynfcFC     | 0.88642+-0.03854 | 0.84757+-0.04583 |
| 63 | transformer             | gcn          | dynfc       | 0.92138+-0.02028 | 0.91679+-0.02225 |
| 64 | graphormer              | gcn          | statfc      | 0.88529+-0.03730 | 0.85005+-0.05132 |
| 65 | graphormer              | gcn          | dynfcFC     | 0.89572+-0.03177 | 0.85674+-0.03720 |
| 66 | graphormer              | gcn          | statfcBOLD  | 0.87492+-0.05187 | 0.85443+-0.04732 |
| 67 | graphormer              | gcn          | statfcFC    | 0.88551+-0.04216 | 0.84773+-0.05245 |
| 68 | none                    | mlp          | dynfcFC     | 0.89320+-0.03180 | 0.86739+-0.04807 |
| 69 | none                    | mlp          | statfcBOLD  | 0.88986+-0.03524 | 0.85268+-0.04820 |
| 70 | none                    | mlp          | dynfcBOLD   | 0.89022+-0.03254 | 0.85020+-0.05028 |
| 71 | none                    | mlp          | statfc      | 0.91597+-0.04441 | 0.90841+-0.05314 |
| 72 | none                    | mlp          | dynfc       | 0.91584+-0.02901 | 0.90825+-0.03796 |
| 73 | none                    | mlp          | statfcFC    | 0.89277+-0.03577 | 0.87053+-0.05002 |


## Data: adni 

|    | backbone                | classifier   | data type      | acc              | f1               |
|---:|:------------------------|:-------------|:---------------|:-----------------|:-----------------|
|  0 | neurodetour4H2L         | gcn          | statfcFC aal   | 0.82000+-0.05103 | 0.78483+-0.07143 |
|  1 | neurodetour4H1L         | gcn          | statfcFC aal   | 0.82667+-0.06320 | 0.78210+-0.08320 |
|  2 | neurodetour4H1L         | gcn          | statfcFC aal   | 0.82000+-0.05103 | 0.77551+-0.06301 |
|  3 | neurodetour3H2L         | gcn          | statfcFC aal   | 0.84963+-0.08934 | 0.81697+-0.11159 |
|  4 | neurodetour6H2L         | gcn          | statfcFC aal   | 0.78370+-0.06228 | 0.74524+-0.10160 |
|  5 | neurodetour6H1L         | gcn          | statfcFC aal   | 0.82815+-0.07899 | 0.78755+-0.08462 |
|  6 | neurodetour5H1L         | gcn          | statfcFC aal   | 0.82815+-0.08724 | 0.79151+-0.10006 |
|  7 | neurodetour5H2L         | gcn          | statfcFC aal   | 0.82741+-0.07433 | 0.78587+-0.09014 |
|  8 | neurodetour3H1L         | gcn          | statfcFC aal   | 0.82148+-0.08967 | 0.77123+-0.10215 |
|  9 | neurodetour             | gcn          | dynfc aal      | 0.85299+-0.04227 | 0.81555+-0.05249 |
| 10 | neurodetour             | gcn          | statfcFC aal   | 0.85556+-0.04969 | 0.83292+-0.04447 |
| 11 | neurodetour1H1L         | gcn          | statfcFC aal   | 0.82889+-0.08664 | 0.79134+-0.10706 |
| 12 | neurodetour1H2L         | gcn          | statfcFC aal   | 0.82074+-0.08222 | 0.78076+-0.09501 |
| 13 | neurodetour7H1L         | gcn          | statfcFC aal   | 0.84222+-0.05222 | 0.78361+-0.06733 |
| 14 | neurodetour2H1L         | gcn          | dynfcBOLD aal  | 0.83675+-0.05643 | 0.78225+-0.07154 |
| 15 | neurodetour2H1L         | gcn          | dynfcFC aal    | 0.83818+-0.03943 | 0.79885+-0.05094 |
| 16 | neurodetour2H1L         | gcn          | statfcFC aal   | 0.79778+-0.06837 | 0.76283+-0.07143 |
| 17 | neurodetour2H1L         | gcn          | statfcBOLD aal | 0.83481+-0.05313 | 0.77953+-0.07671 |
| 18 | transformer             | gcn          | dynfc aal      | 0.83077+-0.05585 | 0.77368+-0.06546 |
| 19 | none                    | gcn          | statfc aal     | 0.80074+-0.10523 | 0.74212+-0.09411 |
| 20 | none                    | gcn          | statfcBOLD aal | 0.80667+-0.07260 | 0.76186+-0.08499 |
| 21 | none                    | gcn          | statfcFC aal   | 0.84222+-0.06917 | 0.78533+-0.09765 |
| 22 | none                    | gcn          | dynfc aal      | 0.83191+-0.05624 | 0.76736+-0.06470 |
| 23 | none                    | gcn          | dynfcFC aal    | 0.83305+-0.06301 | 0.76952+-0.08172 |
| 24 | transformer             | gcn          | statfcBOLD aal | 0.80667+-0.07260 | 0.76800+-0.05944 |
| 25 | transformer             | gcn          | dynfcBOLD aal  | 0.80712+-0.04273 | 0.76693+-0.06895 |
| 26 | transformer             | gcn          | statfcFC aal   | 0.84889+-0.04111 | 0.81154+-0.04827 |
| 27 | transformer             | gcn          | dynfcFC aal    | 0.82422+-0.05984 | 0.78652+-0.07367 |
| 28 | transformer             | gcn          | statfc aal     | 0.81481+-0.10143 | 0.75995+-0.12118 |
| 29 | neurodetourSingleFC2H1L | gcn          | statfcFC aal   | 0.81926+-0.03254 | 0.80973+-0.04196 |
| 30 | neurodetour8H2L         | gcn          | statfcFC aal   | 0.82815+-0.07899 | 0.75565+-0.11674 |
| 31 | neurodetour8H1L         | gcn          | statfcFC aal   | 0.80667+-0.08150 | 0.76852+-0.09679 |
| 32 | neurodetour7H2L         | gcn          | statfcFC aal   | 0.81333+-0.07132 | 0.76666+-0.07460 |
| 33 | none                    | gcn          | dynfcBOLD aal  | 0.83533+-0.05393 | 0.77868+-0.05658 |
| 34 | neurodetourSingleSC2H1L | gcn          | statfcFC aal   | 0.82741+-0.07881 | 0.77505+-0.09385 |
| 35 | neurodetour             | gcn          | dynfcBOLD aal  | 0.81311+-0.05400 | 0.78050+-0.05884 |
| 36 | neurodetour1H1L         | gcn          | statfcFC aal   | 0.81333+-0.08036 | 0.79105+-0.09136 |
| 37 | braingnn                | gcn          | statfc         | 0.82815+-0.07899 | 0.76541+-0.10395 |
| 38 | bolt                    | gcn          | statfc aal     | 0.84296+-0.06402 | 0.81159+-0.08232 |
| 39 | bolt                    | gcn          | dynfc aal      | 0.80541+-0.07010 | 0.78000+-0.06731 |
| 40 | bolt                    | gcn          | statfcBOLD aal | 0.81407+-0.07078 | 0.76680+-0.08774 |
| 41 | braingnn                | gcn          | dynfcFC aal    | 0.83305+-0.05424 | 0.79145+-0.08018 |
| 42 | braingnn                | gcn          | statfcFC aal   | 0.82074+-0.06857 | 0.76566+-0.10013 |
| 43 | braingnn                | gcn          | dynfcBOLD aal  | 0.83419+-0.06048 | 0.78822+-0.06959 |
| 44 | braingnn                | gcn          | statfcBOLD aal | 0.82074+-0.06857 | 0.75106+-0.09689 |
| 45 | bolt                    | gcn          | dynfcFC aal    | 0.80342+-0.02819 | 0.76892+-0.07750 |
| 46 | bnt                     | gcn          | statfc         | 0.82741+-0.05889 | 0.79720+-0.06166 |
| 47 | bnt                     | gcn          | statfcFC aal   | 0.82815+-0.06466 | 0.79679+-0.06149 |
| 48 | bnt                     | gcn          | dynfcBOLD aal  | 0.84330+-0.06992 | 0.80496+-0.08401 |
| 49 | bnt                     | gcn          | statfcBOLD aal | 0.82667+-0.04401 | 0.80164+-0.08009 |
| 50 | bnt                     | gcn          | dynfc          | 0.83903+-0.06037 | 0.80331+-0.07912 |
| 51 | bolt                    | gcn          | dynfcBOLD aal  | 0.80798+-0.07529 | 0.77920+-0.08622 |
| 52 | bolt                    | gcn          | statfcFC aal   | 0.82000+-0.03510 | 0.79639+-0.04328 |
| 53 | bnt                     | gcn          | dynfcFC aal    | 0.83305+-0.06301 | 0.78707+-0.06671 |
| 54 | nagphormer              | gcn          | dynfc aal      | 0.83305+-0.05263 | 0.75789+-0.07409 |
| 55 | nagphormer              | gcn          | statfcBOLD aal | 0.81333+-0.06095 | 0.75400+-0.08582 |
| 56 | nagphormer              | gcn          | statfc aal     | 0.82074+-0.06857 | 0.74119+-0.09557 |
| 57 | nagphormer              | gcn          | dynfcFC aal    | 0.82792+-0.05823 | 0.76464+-0.06935 |
| 58 | neurodetour             | gcn          | statfc aal     | 0.81333+-0.07598 | 0.75608+-0.07678 |
| 59 | neurodetour             | gcn          | dynfcFC aal    | 0.83447+-0.05193 | 0.79931+-0.05829 |
| 60 | neurodetour             | gcn          | statfcBOLD aal | 0.81333+-0.06095 | 0.77346+-0.07355 |
| 61 | nagphormer              | gcn          | dynfcBOLD aal  | 0.82165+-0.05728 | 0.77796+-0.07007 |
| 62 | nagphormer              | gcn          | statfcFC aal   | 0.82741+-0.05889 | 0.76574+-0.06670 |
| 63 | graphormer              | gcn          | statfc aal     | 0.80519+-0.08146 | 0.76016+-0.07528 |
| 64 | graphormer              | gcn          | dynfcFC aal    | 0.83276+-0.05801 | 0.78294+-0.05220 |
| 65 | graphormer              | gcn          | statfcBOLD aal | 0.83481+-0.05313 | 0.77779+-0.05510 |
| 66 | graphormer              | gcn          | dynfcBOLD aal  | 0.81282+-0.06583 | 0.76744+-0.06945 |
| 67 | graphormer              | gcn          | dynfc aal      | 0.84046+-0.05365 | 0.78222+-0.07235 |
| 68 | graphormer              | gcn          | statfcFC aal   | 0.82741+-0.05889 | 0.78136+-0.06027 |
| 69 | none                    | mlp          | statfc aal     | 0.82815+-0.06466 | 0.75678+-0.08937 |
| 70 | none                    | mlp          | statfcBOLD aal | 0.80667+-0.07260 | 0.74956+-0.09167 |
| 71 | none                    | mlp          | dynfc aal      | 0.83305+-0.05263 | 0.76635+-0.08668 |
| 72 | none                    | mlp          | dynfcBOLD aal  | 0.82934+-0.06353 | 0.76746+-0.07081 |
| 73 | none                    | mlp          | dynfcFC aal    | 0.82678+-0.05711 | 0.77979+-0.05790 |
| 74 | none                    | mlp          | statfcFC aal   | 0.79259+-0.10344 | 0.74716+-0.08669 |


## Data: hcpa 

|     | backbone                | classifier   | data type         | acc              | f1               |
|----:|:------------------------|:-------------|:------------------|:-----------------|:-----------------|
|   0 | neurodetour3H2L         | gcn          | statfcFC gordon   | 0.97822+-0.00468 | 0.97825+-0.00463 |
|   1 | neurodetour3H1L         | gcn          | statfcFC gordon   | 0.97224+-0.00187 | 0.97223+-0.00200 |
|   2 | neurodetour4H1L         | gcn          | statfcFC gordon   | 0.97120+-0.00578 | 0.97129+-0.00578 |
|   3 | neurodetour4H2L         | gcn          | statfcFC gordon   | 0.97843+-0.00644 | 0.97838+-0.00645 |
|   4 | neurodetour7H1L         | gcn          | statfcFC gordon   | 0.97861+-0.00228 | 0.97866+-0.00226 |
|   5 | neurodetour6H2L         | gcn          | statfcFC gordon   | 0.97716+-0.00353 | 0.97718+-0.00352 |
|   6 | neurodetour6H1L         | gcn          | statfcFC gordon   | 0.97514+-0.00429 | 0.97513+-0.00428 |
|   7 | neurodetour5H2L         | gcn          | dynfcBOLD gordon  | 0.89814+-0.00688 | 0.89719+-0.00813 |
|   8 | neurodetour5H2L         | gcn          | dynfcFC gordon    | 0.93704+-0.00781 | 0.93657+-0.00764 |
|   9 | neurodetour5H2L         | gcn          | statfcFC gordon   | 0.98233+-0.00454 | 0.98234+-0.00453 |
|  10 | neurodetour5H2L         | gcn          | statfcBOLD gordon | 0.95372+-0.01238 | 0.95376+-0.01252 |
|  11 | neurodetour5H1L         | gcn          | statfcFC gordon   | 0.97346+-0.00684 | 0.97354+-0.00672 |
|  12 | bnt                     | gcn          | dynfcFC aal       | 0.94177+-0.00348 | 0.94156+-0.00346 |
|  13 | neurodetour2H1L         | gcn          | statfcFC aal      | 0.96690+-0.00535 | 0.96696+-0.00543 |
|  14 | neurodetour1H2L         | gcn          | statfcFC gordon   | 0.97328+-0.00341 | 0.97330+-0.00347 |
|  15 | neurodetour1H1L         | gcn          | statfcFC gordon   | 0.96832+-0.00255 | 0.96845+-0.00259 |
|  16 | neurodetour1H1L         | gcn          | statfcFC gordon   | 0.97162+-0.00353 | 0.97162+-0.00356 |
|  17 | neurodetour1H1L         | gcn          | statfcFC gordon   | 0.96998+-0.00427 | 0.96997+-0.00426 |
|  18 | transformer             | gcn          | dynfcFC gordon    | 0.95160+-0.00334 | 0.95134+-0.00348 |
|  19 | neurodetour             | gcn          | statfc aal        | 0.90743+-0.01131 | 0.90732+-0.01086 |
|  20 | neurodetour             | gcn          | dynfc aal         | 0.85982+-0.01232 | 0.85581+-0.01147 |
|  21 | neurodetour             | gcn          | statfcFC aal      | 0.96587+-0.00314 | 0.96581+-0.00308 |
|  22 | neurodetour3H1L         | gcn          | statfcFC gordon   | 0.97533+-0.00275 | 0.97526+-0.00285 |
|  23 | neurodetour2H1L         | gcn          | dynfcBOLD aal     | 0.87540+-0.00770 | 0.87034+-0.00948 |
|  24 | neurodetour2H1L         | gcn          | dynfcFC aal       | 0.92757+-0.00522 | 0.92717+-0.00536 |
|  25 | neurodetour2H1L         | gcn          | dynfcBOLD gordon  | 0.90255+-0.00437 | 0.90214+-0.00448 |
|  26 | neurodetour2H1L         | gcn          | statfcBOLD gordon | 0.97572+-0.00298 | 0.97562+-0.00311 |
|  27 | neurodetour2H1L         | gcn          | statfcBOLD aal    | 0.95026+-0.01927 | 0.95086+-0.01856 |
|  28 | neurodetour             | gcn          | dynfcFC aal       | 0.92163+-0.00415 | 0.92156+-0.00454 |
|  29 | transformer             | gcn          | statfcFC gordon   | 0.97531+-0.00504 | 0.97531+-0.00513 |
|  30 | transformer             | gcn          | dynfc aal         | 0.85091+-0.00963 | 0.84741+-0.00947 |
|  31 | none                    | gcn          | dynfcFC gordon    | 0.94423+-0.00137 | 0.94393+-0.00140 |
|  32 | none                    | gcn          | dynfcFC aal       | 0.91955+-0.00455 | 0.91901+-0.00412 |
|  33 | transformer             | gcn          | dynfcBOLD aal     | 0.89479+-0.00322 | 0.89066+-0.00476 |
|  34 | transformer             | gcn          | statfcFC aal      | 0.96876+-0.00592 | 0.96891+-0.00588 |
|  35 | transformer             | gcn          | dynfcBOLD gordon  | 0.90015+-0.00395 | 0.89881+-0.00372 |
|  36 | transformer             | gcn          | statfcBOLD aal    | 0.90182+-0.05286 | 0.90210+-0.05231 |
|  37 | transformer             | gcn          | dynfc gordon      | 0.88452+-0.00307 | 0.88555+-0.00299 |
|  38 | transformer             | gcn          | statfc aal        | 0.90186+-0.01015 | 0.90189+-0.00990 |
|  39 | transformer             | gcn          | statfc gordon     | 0.92120+-0.00781 | 0.92159+-0.00800 |
|  40 | transformer             | gcn          | statfcBOLD gordon | 0.97266+-0.01371 | 0.97247+-0.01395 |
|  41 | transformer             | gcn          | dynfcFC aal       | 0.92238+-0.00310 | 0.92224+-0.00393 |
|  42 | neurodetour7H2L         | gcn          | statfcFC gordon   | 0.97923+-0.00308 | 0.97932+-0.00307 |
|  43 | none                    | gcn          | statfcBOLD gordon | 0.97427+-0.00508 | 0.97418+-0.00511 |
|  44 | neurodetourSingleSC2H1L | gcn          | statfcFC aal      | 0.96813+-0.00721 | 0.96824+-0.00731 |
|  45 | neurodetourSingleFC2H1L | gcn          | statfcFC aal      | 0.96875+-0.00410 | 0.96895+-0.00408 |
|  46 | neurodetourSingleFC2H1L | gcn          | statfcFC gordon   | 0.97717+-0.00342 | 0.97720+-0.00341 |
|  47 | neurodetourSingleSC2H1L | gcn          | statfcFC gordon   | 0.97326+-0.00444 | 0.97339+-0.00434 |
|  48 | neurodetour8H1L         | gcn          | statfcFC gordon   | 0.97428+-0.00441 | 0.97440+-0.00435 |
|  49 | neurodetour8H2L         | gcn          | statfcFC gordon   | 0.97964+-0.00299 | 0.97963+-0.00304 |
|  50 | none                    | gcn          | statfcBOLD aal    | 0.92944+-0.00579 | 0.92979+-0.00603 |
|  51 | neurodetourSingleSC2H1L | gcn          | statfcFC aal      | 0.96689+-0.00563 | 0.96695+-0.00561 |
|  52 | none                    | gcn          | statfc aal        | 0.90500+-0.00912 | 0.90485+-0.00938 |
|  53 | none                    | gcn          | statfcFC gordon   | 0.96668+-0.00385 | 0.96674+-0.00406 |
|  54 | none                    | gcn          | statfc gordon     | 0.92202+-0.00888 | 0.92193+-0.00916 |
|  55 | none                    | gcn          | dynfc aal         | 0.85213+-0.00590 | 0.84847+-0.00777 |
|  56 | none                    | gcn          | dynfc gordon      | 0.88258+-0.00739 | 0.88194+-0.00751 |
|  57 | none                    | gcn          | statfcFC aal      | 0.95848+-0.00931 | 0.95852+-0.00951 |
|  58 | none                    | gcn          | dynfcBOLD aal     | 0.84597+-0.00452 | 0.83945+-0.00365 |
|  59 | none                    | gcn          | dynfcBOLD gordon  | 0.88294+-0.00278 | 0.88206+-0.00317 |
|  60 | neurodetour             | gcn          | dynfcBOLD aal     | 0.89168+-0.00355 | 0.88950+-0.00500 |
|  61 | bolt                    | gcn          | statfcFC gordon   | 0.95192+-0.02558 | 0.95236+-0.02504 |
|  62 | braingnn                | gcn          | dynfcBOLD aal     | 0.72616+-0.03327 | 0.64396+-0.06670 |
|  63 | bolt                    | gcn          | dynfc aal         | 0.85024+-0.00483 | 0.84465+-0.00564 |
|  64 | bolt                    | gcn          | statfc aal        | 0.88585+-0.00694 | 0.88587+-0.00723 |
|  65 | bolt                    | gcn          | dynfcFC aal       | 0.91679+-0.00375 | 0.91662+-0.00386 |
|  66 | bolt                    | gcn          | statfcBOLD aal    | 0.95783+-0.00547 | 0.95782+-0.00568 |
|  67 | braingnn                | gcn          | dynfcBOLD gordon  | 0.77779+-0.01735 | 0.75968+-0.04101 |
|  68 | braingnn                | gcn          | statfcFC aal      | 0.90849+-0.01354 | 0.90923+-0.01414 |
|  69 | braingnn                | gcn          | statfcFC gordon   | 0.91531+-0.03106 | 0.91503+-0.03000 |
|  70 | braingnn                | gcn          | dynfcFC aal       | 0.86058+-0.02645 | 0.85427+-0.03374 |
|  71 | braingnn                | gcn          | statfc gordon     | 0.78305+-0.02389 | 0.77707+-0.02450 |
|  72 | braingnn                | gcn          | statfc aal        | 0.73298+-0.12699 | 0.67655+-0.20136 |
|  73 | braingnn                | gcn          | dynfc gordon      | 0.75492+-0.09648 | 0.67576+-0.16355 |
|  74 | braingnn                | gcn          | statfcBOLD aal    | 0.89379+-0.02876 | 0.89382+-0.02922 |
|  75 | bnt                     | gcn          | dynfc aal         | 0.86681+-0.00584 | 0.86252+-0.00647 |
|  76 | bnt                     | gcn          | dynfcBOLD gordon  | 0.90415+-0.01375 | 0.90340+-0.01465 |
|  77 | bnt                     | gcn          | dynfcBOLD aal     | 0.86548+-0.00370 | 0.86446+-0.00405 |
|  78 | bnt                     | gcn          | statfc aal        | 0.91116+-0.00639 | 0.91064+-0.00679 |
|  79 | bnt                     | gcn          | statfcBOLD gordon | 0.94613+-0.00723 | 0.94537+-0.00758 |
|  80 | bnt                     | gcn          | dynfc gordon      | 0.92386+-0.00791 | 0.92334+-0.00892 |
|  81 | bnt                     | gcn          | statfc gordon     | 0.93979+-0.01648 | 0.93936+-0.01721 |
|  82 | braingnn                | gcn          | statfcBOLD gordon | 0.80098+-0.03503 | 0.77252+-0.05607 |
|  83 | bolt                    | gcn          | dynfcBOLD aal     | 0.91925+-0.00689 | 0.91763+-0.00778 |
|  84 | bolt                    | gcn          | statfcFC aal      | 0.96402+-0.00411 | 0.96380+-0.00425 |
|  85 | bolt                    | gcn          | statfcBOLD gordon | 0.97061+-0.00306 | 0.97059+-0.00309 |
|  86 | bolt                    | gcn          | dynfcBOLD gordon  | 0.95755+-0.00367 | 0.95759+-0.00349 |
|  87 | bolt                    | gcn          | dynfcFC gordon    | 0.93817+-0.01220 | 0.93853+-0.01186 |
|  88 | bnt                     | gcn          | dynfcFC gordon    | 0.92866+-0.00680 | 0.92867+-0.00751 |
|  89 | bnt                     | gcn          | statfcFC gordon   | 0.95387+-0.01011 | 0.95348+-0.01037 |
|  90 | bnt                     | gcn          | statfcFC aal      | 0.97922+-0.00651 | 0.97918+-0.00658 |
|  91 | bnt                     | gcn          | statfcBOLD aal    | 0.92570+-0.01189 | 0.92575+-0.01223 |
|  92 | nagphormer              | gcn          | statfcBOLD aal    | 0.94756+-0.01154 | 0.94758+-0.01162 |
|  93 | nagphormer              | gcn          | dynfc aal         | 0.83279+-0.00511 | 0.82692+-0.00702 |
|  94 | nagphormer              | gcn          | dynfcFC aal       | 0.90728+-0.00635 | 0.90637+-0.00683 |
|  95 | nagphormer              | gcn          | statfc gordon     | 0.90227+-0.01151 | 0.90122+-0.01278 |
|  96 | nagphormer              | gcn          | statfcBOLD gordon | 0.96584+-0.00568 | 0.96588+-0.00585 |
|  97 | nagphormer              | gcn          | dynfcFC gordon    | 0.93785+-0.00624 | 0.93752+-0.00613 |
|  98 | nagphormer              | gcn          | statfc aal        | 0.88195+-0.00959 | 0.88150+-0.00929 |
|  99 | nagphormer              | gcn          | statfcFC gordon   | 0.96524+-0.00569 | 0.96548+-0.00564 |
| 100 | neurodetour             | gcn          | dynfcBOLD gordon  | 0.90027+-0.00329 | 0.89875+-0.00398 |
| 101 | neurodetour             | gcn          | statfcBOLD aal    | 0.94323+-0.01115 | 0.94355+-0.01064 |
| 102 | neurodetour             | gcn          | statfcBOLD gordon | 0.96960+-0.00813 | 0.96945+-0.00802 |
| 103 | neuroDetour3Head        | gcn          | statfc gordon     | 0.93476+-0.01067 | 0.93472+-0.01100 |
| 104 | graphormer              | gcn          | statfcFC aal      | 0.78796+-0.05887 | 0.77286+-0.07152 |
| 105 | nagphormer              | gcn          | dynfcBOLD gordon  | 0.86889+-0.00490 | 0.86698+-0.00435 |
| 106 | nagphormer              | gcn          | dynfcBOLD aal     | 0.82021+-0.01770 | 0.81058+-0.02035 |
| 107 | nagphormer              | gcn          | statfcFC aal      | 0.93668+-0.00959 | 0.93685+-0.00949 |
| 108 | graphormer              | gcn          | statfcFC gordon   | 0.77763+-0.13693 | 0.78150+-0.12425 |
| 109 | graphormer              | gcn          | dynfcBOLD aal     | 0.65009+-0.03843 | 0.57040+-0.01459 |
| 110 | graphormer              | gcn          | statfcBOLD gordon | 0.75029+-0.03134 | 0.72688+-0.05497 |
| 111 | graphormer              | gcn          | statfcBOLD aal    | 0.59629+-0.06070 | 0.53052+-0.05813 |
| 112 | graphormer              | gcn          | dynfcBOLD gordon  | 0.70320+-0.00733 | 0.63767+-0.02682 |
| 113 | graphormer              | gcn          | statfc aal        | 0.70355+-0.04671 | 0.68297+-0.05623 |
| 114 | braingnn                | gcn          | dynfc aal         | 0.77864+-0.05488 | 0.73315+-0.10367 |
| 115 | graphormer              | gcn          | dynfcFC aal       | 0.78726+-0.01906 | 0.75256+-0.02659 |
| 116 | none                    | mlp          | statfc gordon     | 0.58463+-0.00265 | 0.43139+-0.00320 |
| 117 | none                    | mlp          | dynfc gordon      | 0.68770+-0.00430 | 0.56046+-0.00558 |
| 118 | none                    | mlp          | statfcFC gordon   | 0.96914+-0.00501 | 0.96903+-0.00507 |
| 119 | none                    | mlp          | dynfcBOLD aal     | 0.83638+-0.01331 | 0.82862+-0.01677 |
| 120 | none                    | mlp          | statfc aal        | 0.58401+-0.00189 | 0.43151+-0.00340 |
| 121 | none                    | mlp          | statfcFC aal      | 0.96010+-0.00496 | 0.96012+-0.00493 |
| 122 | none                    | mlp          | dynfc aal         | 0.68709+-0.00382 | 0.56055+-0.00569 |
| 123 | none                    | mlp          | dynfcFC gordon    | 0.95177+-0.00512 | 0.95190+-0.00506 |
| 124 | none                    | mlp          | dynfcFC aal       | 0.92566+-0.00318 | 0.92518+-0.00354 |
| 125 | none                    | mlp          | statfcBOLD gordon | 0.97638+-0.00706 | 0.97633+-0.00697 |
| 126 | none                    | mlp          | dynfcBOLD gordon  | 0.89064+-0.00441 | 0.89004+-0.00314 |
| 127 | none                    | mlp          | statfcBOLD aal    | 0.93418+-0.00578 | 0.93425+-0.00576 |
