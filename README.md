# NLP_Assignment2

The sampled dataset and tokenizer files are uploaded on onedrive: https://shorturl.at/jz8V1

## Tokenizer Training:

| Dataset Size (in MBs) | Fertility Score | Tokenizer   |
| --------------------- | --------------- | ----------- |
| 500                   | 1.062340199     | Tokenizer 1 |
| 800                   | 1.061760271     | Tokenizer 2 |
| 1000                  | 1.061677823     | Tokenizer 3 |
| 1200                  | 1.061477151     | Tokenizer 4 |
| 1500                  | 1.061219034     | Tokenizer 5 |

## Model Training
| epoch | learning_rate | loss | grad_norm | eval_loss | perplexity |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.00198 | 7.8823 | 10.83006763458252 | 5.341874122619629 | 208.903855 |
| 2 | 0.00196 | 4.5232 | 4.135104656219482 | 5.084144115447998 | 161.441705 |
| 3 | 0.0019399999999999999 | 4.3328 | 5.754868030548096 | 5.077797889709473 | 160.420403 |
| 4 | 0.00192 | 4.2184 | 1.5312623977661133 | 5.037759304046631 | 154.124282 |
| 5 | 0.0019 | 4.0415 | 3.030031204223633 | 4.690997123718262 | 108.961774 |
| 6 | 0.00188 | 3.7058 | 0.3274911642074585 | 4.8733367919921875 | 130.756497 |
| 7 | 0.00186 | 3.6343 | 3.964893102645874 | 4.828609943389893 | 125.037031 |
| 8 | 0.00184 | 3.5964 | 0.43179377913475037 | 4.660287857055664 | 105.666495 |
| 9 | 0.00182 | 3.5288 | 1.0899018049240112 | 4.745774269104004 | 115.096887 |
| 10 | 0.0018000000000000002 | 3.5043 | 0.31293967366218567 | 4.746968746185303 | 115.23445 |
| 11 | 0.0017800000000000001 | 3.4705 | 0.3022761344909668 | 4.784313678741455 | 119.619238 |
| 12 | 0.00176 | 3.4239 | 0.2797553837299347 | 4.777924537658691 | 118.85741 |
| 13 | 0.00174 | 3.372 | 0.27372387051582336 | 4.916806221008301 | 136.565755 |
| 14 | 0.00172 | 3.3172 | 0.3576394319534302 | 4.782943248748779 | 119.45542 |
| 15 | 0.0017 | 3.2378 | 0.3370823264122009 | 4.842909812927246 | 126.83789 |
| 16 | 0.00168 | 3.1721 | 0.4582008719444275 | 4.923586368560791 | 137.494837 |
| 17 | 0.00166 | 3.1168 | 0.411830335855484 | 4.909565448760986 | 135.580485 |
| 18 | 0.00164 | 3.0482 | 0.36471500992774963 | 4.934050559997559 | 138.941164 |
| 19 | 0.0016200000000000001 | 2.9746 | 0.3295668959617615 | 4.928074359893799 | 138.1133 |
| 20 | 0.0016 | 2.8831 | 0.34055638313293457 | 4.919163703918457 | 136.888086 |
| 21 | 0.00158 | 2.7942 | 2.326063632965088 | 4.925785541534424 | 137.797545 |
| 22 | 0.0015600000000000002 | 2.7186 | 0.4043227434158325 | 4.951541423797607 | 141.392742 |
| 23 | 0.0015400000000000001 | 2.6065 | 1.0006740093231201 | 4.939196586608887 | 139.658001 |
| 24 | 0.00152 | 2.546 | 0.6557777523994446 | 4.991822242736816 | 147.204421 |
| 25 | 0.0015 | 2.4463 | 0.8707776665687561 | 5.037770748138428 | 154.126046 |
| 26 | 0.00148 | 2.3249 | 0.4661215543746948 | 4.9997029304504395 | 148.369077 |
| 27 | 0.00146 | 2.2049 | 0.4977121651172638 | 5.0664215087890625 | 158.605741 |
| 28 | 0.0014399999999999999 | 2.0833 | 1.2101680040359497 | 5.072560787200928 | 159.582461 |
| 29 | 0.00142 | 1.9741 | 0.50432950258255 | 5.067237377166748 | 158.735196 |
| 30 | 0.0014 | 1.847 | 0.5038468241691589 | 5.123378276824951 | 167.901631 |
| 31 | 0.00138 | 1.7202 | 0.48927393555641174 | 5.142223358154297 | 171.095753 |
| 32 | 0.00136 | 1.5944 | 0.5299854278564453 | 5.186912536621094 | 178.915305 |
| 33 | 0.00134 | 1.4848 | 0.8718563914299011 | 5.246255397796631 | 189.854008 |
| 34 | 0.00132 | 1.3981 | 0.6798862218856812 | 5.223137378692627 | 185.515304 |
| 35 | 0.0013000000000000002 | 1.2692 | 0.5268112421035767 | 5.28732442855835 | 197.813453 |
| 36 | 0.00128 | 1.158 | 0.682898223400116 | 5.247048854827881 | 190.004709 |
| 37 | 0.00126 | 1.0436 | 0.7518463730812073 | 5.378556251525879 | 216.709176 |
| 38 | 0.00124 | 0.9438 | 0.6754593849182129 | 5.35942268371582 | 212.602172 |
| 39 | 0.00122 | 0.856 | 0.6998625993728638 | 5.463707447052002 | 235.970653 |
| 40 | 0.0012 | 0.7674 | 0.7664692401885986 | 5.458903789520264 | 234.839849 |
| 41 | 0.00118 | 0.6858 | 0.7484133839607239 | 5.474850654602051 | 238.614828 |
| 42 | 0.00116 | 0.6037 | 0.7357663512229919 | 5.530375957489014 | 252.238724 |
| 43 | 0.00114 | 0.5423 | 0.9653528928756714 | 5.5090155601501465 | 246.907941 |
| 44 | 0.0011200000000000001 | 0.497 | 0.612908661365509 | 5.572904109954834 | 263.197344 |
| 45 | 0.0011 | 0.4377 | 0.5782804489135742 | 5.601387023925781 | 270.801756 |
| 46 | 0.00108 | 0.3905 | 0.571628212928772 | 5.680751323699951 | 293.169612 |
| 47 | 0.0010600000000000002 | 0.3414 | 0.6290593147277832 | 5.737216949462891 | 310.199907 |
| 48 | 0.0010400000000000001 | 0.3011 | 0.6807319521903992 | 5.684469223022461 | 294.261616 |
| 49 | 0.00102 | 0.2694 | 0.4352017939090729 | 5.724931240081787 | 306.412196 |
| 50 | 0.001 | 0.2469 | 0.45147380232810974 | 5.81417179107666 | 335.013822 |
| 51 | 0.00098 | 0.2316 | 0.39783775806427 | 5.801687240600586 | 330.857325 |
| 52 | 0.00096 | 0.2169 | 0.43664655089378357 | 5.922039031982422 | 373.171848 |
| 53 | 0.00094 | 0.2025 | 0.2688675820827484 | 5.842624664306641 | 344.682831 |
| 54 | 0.00092 | 0.1882 | 0.1963178813457489 | 5.89838171005249 | 364.447209 |
| 55 | 0.0009000000000000001 | 0.1767 | 0.14648126065731049 | 5.9089179039001465 | 368.307396 |
| 56 | 0.00088 | 0.1707 | 0.1374797821044922 | 5.998571395874023 | 402.852865 |
| 57 | 0.00086 | 0.1645 | 0.09939230233430862 | 5.998988628387451 | 403.020983 |
| 58 | 0.00084 | 0.1597 | 0.0886126309633255 | 6.020808696746826 | 411.911573 |
| 59 | 0.00082 | 0.156 | 0.06901884824037552 | 6.048976898193359 | 423.679341 |
| 60 | 0.0008 | 0.154 | 0.07904954999685287 | 6.053466796875 | 425.585895 |
| 61 | 0.0007800000000000001 | 0.1521 | 0.08204028010368347 | 6.08182430267334 | 437.827196 |
| 62 | 0.00076 | 0.1508 | 0.08373957127332687 | 6.086550235748291 | 439.901235 |
| 63 | 0.00074 | 0.1495 | 0.06548644602298737 | 6.092987537384033 | 442.742146 |
| 64 | 0.0007199999999999999 | 0.1489 | 0.0749504417181015 | 6.108479976654053 | 449.65471 |
| 65 | 0.0007 | 0.1491 | 0.06666584312915802 | 6.119835376739502 | 454.789819 |
| 66 | 0.00068 | 0.1479 | 0.090924933552742 | 6.116401195526123 | 453.230667 |
| 67 | 0.00066 | 0.147 | 0.0627363845705986 | 6.120159149169922 | 454.937092 |
| 68 | 0.00064 | 0.1464 | 0.06087152659893036 | 6.132468223571777 | 460.571552 |
| 69 | 0.00062 | 0.1458 | 0.06984956562519073 | 6.151335716247559 | 469.343879 |
| 70 | 0.0006 | 0.1452 | 0.050487469881772995 | 6.143265247344971 | 465.571297 |
| 71 | 0.00058 | 0.1448 | 0.055137746036052704 | 6.14772891998291 | 467.6541 |
| 72 | 0.0005600000000000001 | 0.1445 | 0.058908458799123764 | 6.147255897521973 | 467.432941 |
| 73 | 0.00054 | 0.1439 | 0.06310687214136124 | 6.15134859085083 | 469.349921 |
| 74 | 0.0005200000000000001 | 0.1438 | 0.06702357530593872 | 6.178257465362549 | 482.151059 |
| 75 | 0.0005 | 0.1436 | 0.06692282110452652 | 6.178020477294922 | 482.036808 |
| 76 | 0.00048 | 0.1431 | 0.05657191947102547 | 6.161510467529297 | 474.143713 |
| 77 | 0.00046 | 0.1429 | 0.06557995826005936 | 6.1723480224609375 | 479.310217 |
| 78 | 0.00044 | 0.1426 | 0.057419553399086 | 6.171828269958496 | 479.061159 |
| 79 | 0.00042 | 0.1425 | 0.050767987966537476 | 6.180866718292236 | 483.410756 |
| 80 | 0.0004 | 0.1422 | 0.06195845082402229 | 6.181358337402344 | 483.648468 |
| 81 | 0.00038 | 0.1422 | 0.054841767996549606 | 6.179361343383789 | 482.683589 |
| 82 | 0.00035999999999999997 | 0.1416 | 0.0683060735464096 | 6.187324523925781 | 486.54263 |
| 83 | 0.00034 | 0.1418 | 0.05761469528079033 | 6.189059734344482 | 487.387617 |
| 84 | 0.00032 | 0.1418 | 0.06079474464058876 | 6.187809467315674 | 486.778633 |
| 85 | 0.0003 | 0.141 | 0.07209662348031998 | 6.194770336151123 | 490.178856 |
| 86 | 0.00028000000000000003 | 0.1413 | 0.08093111217021942 | 6.190364360809326 | 488.023891 |
| 87 | 0.00026000000000000003 | 0.1409 | 0.05694061145186424 | 6.195146083831787 | 490.363074 |
| 88 | 0.00024 | 0.1409 | 0.05375082045793533 | 6.202808856964111 | 494.135048 |
| 89 | 0.00022 | 0.1407 | 0.05568912997841835 | 6.201825141906738 | 493.649199 |
| 90 | 0.0002 | 0.1403 | 0.05441686883568764 | 6.201447486877441 | 493.462805 |
| 91 | 0.00017999999999999998 | 0.1406 | 0.04904496297240257 | 6.201365947723389 | 493.42257 |
| 92 | 0.00016 | 0.1403 | 0.04608221724629402 | 6.206197738647461 | 495.812454 |
| 93 | 0.00014000000000000001 | 0.1402 | 0.048730384558439255 | 6.204916477203369 | 495.177596 |
| 94 | 0.00012 | 0.1401 | 0.04741162806749344 | 6.207217693328857 | 496.318418 |
| 95 | 0.0001 | 0.1399 | 0.06511170417070389 | 6.20723295211792 | 496.325992 |
| 96 | 8.0 | 0.1401 | 0.04602372646331787 | 6.207551956176758 | 496.484347 |
| 97 | 6.0 | 0.1399 | 0.05116153135895729 | 6.209137916564941 | 497.272376 |
| 98 | 4.0 | 0.1399 | 0.07090374082326889 | 6.210533142089844 | 497.966668 |
| 99 | 2.0 | 0.1397 | 0.04762996360659599 | 6.210764408111572 | 498.081844 |
| 100 |  0 |  0.1398 |  0.05096082389354706 |  - |  - |

### Individual Contributions of Group
| Name             | Roll No. | Contribution                                                                       |
| ---------------- | -------- | ---------------------------------------------------------------------------------- |
| Husain Malwat    | 21110117 | Contributed in data sampling, tokenizer training and model parameters calculation. |
| Amey Rangari     | 21110177 | Contributed in Model pretraining, perplexity values calculations.                  |
| Netram Choudhary | 21110138 | Contributed in training the tokenizer.                                             |
| Vinay Goud       | 21110125 | Contributed in training the tokenizer and model pretraining.                       |
| Dhruv Patel      | 23210035 | Contributed in Model pretraining, perplexity values calculations.                  |
