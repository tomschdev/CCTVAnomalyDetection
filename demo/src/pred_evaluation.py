import time
import numpy as np
import os
import sys
from functools import reduce
from random import shuffle
import copy
import time
from itertools import islice
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import streamlit as st
import base64
from itertools import islice
from statsmodels.tsa.seasonal import seasonal_decompose
import altair as alt
from math import ceil
from PIL import Image
from streamlit_player import st_player

root=os.getcwd()
ANNO_FILE=os.path.join(root, "demo", "annotations", "Temporal_Anomaly_Annotation.txt")
min_thresh = 0.1
max_thresh = 0.9
CLASSES = ["A", "N"]
PER=10

download_map = {
    "Vandalism/036" : "https://drive.google.com/file/d/17Ux1IqYHxtWXEF9b6_88ptDKPSBExuko/view?usp=drivesdk", 	               
    "Vandalism/028" : "https://drive.google.com/file/d/17R5QI0ZkMslJYtSIxeZkIF2OskXL4or3/view?usp=drivesdk", 	
    "Vandalism/017" : "https://drive.google.com/file/d/17OBr6u4P2T7isjAwLkxVfQLnhjIfs-gj/view?usp=drivesdk", 	
    "Vandalism/015" : "https://drive.google.com/file/d/17MNaLMqiLzV9iRgb59s397uS59samGjo/view?usp=drivesdk", 	
    "Vandalism/007" : "https://drive.google.com/file/d/17I5-R7DKuGLi7wcHng4AkM9iwPfpUmWL/view?usp=drivesdk", 	
    "Stealing/079" : "https://drive.google.com/file/d/17FyTR1mpqjkihJJkHQJhD5Cu6q9G01OT/view?usp=drivesdk", 	
    "Stealing/062" : "https://drive.google.com/file/d/172Q5Q8jGtUDBqvtEGALomdEW9grnhXc-/view?usp=drivesdk", 	
    "Stealing/058" : "https://drive.google.com/file/d/16mfBdGp7iLvAa7hPD9Z3uS5E-dNasl5P/view?usp=drivesdk", 	
    "Stealing/036" : "https://drive.google.com/file/d/16d0xZz7hYW4fo6M0fy3MlX3OfJfjeBav/view?usp=drivesdk", 	
    "Stealing/019" : "https://drive.google.com/file/d/16ZAQJJLf9_mUO9LEAatCp4DbAYA8ZO5s/view?usp=drivesdk", 	
    "Shoplifting/049" : "https://drive.google.com/file/d/16WqGz8b0xsR_0h9cGXYZ5eiQ6RSg74v4/view?usp=drivesdk", 	
    "Shoplifting/044" : "https://drive.google.com/file/d/16W30r8P6geRDygqqI4LkhPAVnTAL2O_N/view?usp=drivesdk", 	
    "Shoplifting/039" : "https://drive.google.com/file/d/16TIaAM-87wHG-diYgvHlE-IauaY914C1/view?usp=drivesdk", 	
    "Shoplifting/037" : "https://drive.google.com/file/d/16SC_4md1QGed4klbgTMNpd_U1WtlzFQg/view?usp=drivesdk", 	
    "Shoplifting/034" : "https://drive.google.com/file/d/16RYMGjTRP_m9We1lMVpm_x75-SE7lq4y/view?usp=drivesdk", 	
    "Shoplifting/033" : "https://drive.google.com/file/d/16P8XrYKKQBonxCyMcUrKWx5wZtxmeqe6/view?usp=drivesdk", 	
    "Shoplifting/031" : "https://drive.google.com/file/d/16Nc8OX_T9VSpTrPKxHLX0HV3z0NGQGds/view?usp=drivesdk", 	
    "Shoplifting/029" : "https://drive.google.com/file/d/16MAZVnDh1qcmgqpTDzC5JhfVtCUp1HTU/view?usp=drivesdk", 	
    "Shoplifting/028" : "https://drive.google.com/file/d/16GLrFMDt4EiXx_r49XjgdCbTi_8MM6nm/view?usp=drivesdk", 	
    "Shoplifting/027" : "https://drive.google.com/file/d/16DJX9_KwMe0oq1EN8DOkVEzi83X-KF0T/view?usp=drivesdk", 	
    "Shoplifting/022" : "https://drive.google.com/file/d/166uqo2TmHG3V3du6Aa4WtVy6UJm-D37B/view?usp=drivesdk", 	
    "Shoplifting/021" : "https://drive.google.com/file/d/164g1zM3PQSYHeMmG8JV5Pd7nyn1bCyqK/view?usp=drivesdk", 	
    "Shoplifting/020" : "https://drive.google.com/file/d/163swqncqsQbDZnWj1jZqtO1yInTEUe31/view?usp=drivesdk", 	
    "Shoplifting/017" : "https://drive.google.com/file/d/163YUfNk_cpbTavdaZXzMrHvjxfOCszrM/view?usp=drivesdk", 	
    "Shoplifting/016" : "https://drive.google.com/file/d/15uKhzDb0zCXQ8npXY2WYzE9ksexVwCN1/view?usp=drivesdk", 	
    "Shoplifting/015" : "https://drive.google.com/file/d/15t0VYtsMPKsfbvlUMdF4tD4h7pDyc8r1/view?usp=drivesdk", 	
    "Shoplifting/010" : "https://drive.google.com/file/d/15r07D73ClkRmpxEY7AK2UVwwnUzz9a2P/view?usp=drivesdk", 	
    "Shoplifting/007" : "https://drive.google.com/file/d/15lNgEGMslVJhtH7mipdZiA0eXeDEdqrh/view?usp=drivesdk", 	
    "Shoplifting/005" : "https://drive.google.com/file/d/15hkYORbDPkcrZ0sXaKxAQdGRJ3w98T6p/view?usp=drivesdk", 	
    "Shoplifting/004" : "https://drive.google.com/file/d/15hY0M9kI8OYreGRUV9LA63L3acYpZTuZ/view?usp=drivesdk", 	
    "Shoplifting/001" : "https://drive.google.com/file/d/15ZWaux1gcaBg8L4PWiuHM0HwHo5y99aU/view?usp=drivesdk", 	
    "Shooting/048" : "https://drive.google.com/file/d/15c69fVHJSioFYxirHScraAR9xgFbaMy0/view?usp=drivesdk", 	
    "Shooting/047" : "https://drive.google.com/file/d/15VzkOl5Tl-3YjDdZ9XnU-uKOwp19wDSg/view?usp=drivesdk", 	
    "Shooting/046" : "https://drive.google.com/file/d/15RdlmP_Za7TZtBvJ8qi-027uTE_UoAVD/view?usp=drivesdk", 	
    "Shooting/043" : "https://drive.google.com/file/d/15RUW0uuyXeXWHZskHqeiH4P152CF_jUR/view?usp=drivesdk", 	
    "Shooting/037" : "https://drive.google.com/file/d/15PdMrzpEY_JEwIwvGg4QaaNnWfVTAqbV/view?usp=drivesdk", 	
    "Shooting/034" : "https://drive.google.com/file/d/15LivepKiZJk-dVYwrgFD1P6ehgxcpgg7/view?usp=drivesdk", 	
    "Shooting/033" : "https://drive.google.com/file/d/15P8lLPJG9zKimVf8Gx8HhPnxd0Lz3Kqr/view?usp=drivesdk", 	
    "Shooting/032" : "https://drive.google.com/file/d/15Kh-fV9H8MLRiP0CFNDeRbv5d4qS7vmb/view?usp=drivesdk", 	
    "Shooting/028" : "https://drive.google.com/file/d/15IrDO9BGmlbQTGy2mtj8uN29-N7i7pQ8/view?usp=drivesdk", 	
    "Shooting/026" : "https://drive.google.com/file/d/15Bhng92ToMMPkS2_fCF7LSSfGq1Kdl-w/view?usp=drivesdk", 	
    "Shooting/024" : "https://drive.google.com/file/d/15844OY32orOGuqkO8MBVQGtFEU7pcMHl/view?usp=drivesdk", 	
    "Shooting/022" : "https://drive.google.com/file/d/157Y8I_99z7hnj649B36UZzqnCdEQrVbL/view?usp=drivesdk", 	
    "Shooting/021" : "https://drive.google.com/file/d/1574FWgtDBqlk0_7kYzHhcnGivAHfr07n/view?usp=drivesdk", 	
    "Shooting/019" : "https://drive.google.com/file/d/155vp8bCkjKcQXvnaJ870_eATAqfMWWHC/view?usp=drivesdk", 	
    "Shooting/018" : "https://drive.google.com/file/d/156RbvyqkQs9NDyeeTQuzCxL2qvFq8kBH/view?usp=drivesdk", 	
    "Shooting/015" : "https://drive.google.com/file/d/1520jVuxdD7AUk2H9utGKGYtMJ4ILX7E9/view?usp=drivesdk", 	
    "Shooting/013" : "https://drive.google.com/file/d/151tx2c5wrN93ZOSfHa0iIGPdGADgC3uh/view?usp=drivesdk", 	
    "Shooting/011" : "https://drive.google.com/file/d/14yX-WDjSoiTida9eq3JMtHfpCdUXHGhE/view?usp=drivesdk", 	
    "Shooting/010" : "https://drive.google.com/file/d/14w8-Ai3DPrJrgz9XkPYSecY06U9CK6lM/view?usp=drivesdk", 	
    "Shooting/008" : "https://drive.google.com/file/d/14sDXGaVnXVKOvoFBNZePC2ssWKDolFjb/view?usp=drivesdk", 	
    "Shooting/007" : "https://drive.google.com/file/d/14bNCouDRD2fMs6Y9x4ktAIdEkjPDbEEW/view?usp=drivesdk", 	
    "Shooting/004" : "https://drive.google.com/file/d/14_j04fJn38beqtSnbvkVktLyZ67IEynP/view?usp=drivesdk", 	
    "Shooting/002" : "https://drive.google.com/file/d/14Wjshpy1FRtYn7GMO7lxUzNY25QvJpXi/view?usp=drivesdk", 	
    "Robbery/137" : "https://drive.google.com/file/d/14V8lNaQoNn6JEvdgJ4NWqxLulst5oFEK/view?usp=drivesdk", 	
    "Robbery/106" : "https://drive.google.com/file/d/14StJ7vKF24rTkNyA1vIz7iUIAoPuwZvN/view?usp=drivesdk", 	
    "Robbery/102" : "https://drive.google.com/file/d/14OJKpCTB69fzpdqx_7AuA26uy4MchmUN/view?usp=drivesdk", 	
    "Robbery/050" : "https://drive.google.com/file/d/14INj_Q_drHs2Zow4P1zixer9GIcANJwq/view?usp=drivesdk", 	
    "Robbery/048" : "https://drive.google.com/file/d/14GqlyxNeGf7jToy5Y4KnVtRdg2nbvaJx/view?usp=drivesdk", 	
    "RoadAccidents/133" : "https://drive.google.com/file/d/14FvWkRS0o5IzBS5bk1sVacfwbFjFrJsZ/view?usp=drivesdk", 	
    "RoadAccidents/132" : "https://drive.google.com/file/d/14Cij4O4fwTb0CJ3BM4FRhAmEpJOhLgIS/view?usp=drivesdk", 	
    "RoadAccidents/131" : "https://drive.google.com/file/d/145ykTgAvFwJV4NITenu7lMZ6__sIGB06/view?usp=drivesdk", 	
    "RoadAccidents/128" : "https://drive.google.com/file/d/142kahtXgWhq4-0tJTwNF5as1r-h-rmS7/view?usp=drivesdk", 	
    "RoadAccidents/127" : "https://drive.google.com/file/d/140TmuD7FynWybbGl7tsx8mo6rjlZYM1v/view?usp=drivesdk", 	
    "RoadAccidents/125" : "https://drive.google.com/file/d/13sOox0W0PhKEEUScG9qhgtnISM_B1Y0J/view?usp=drivesdk", 	
    "RoadAccidents/124" : "https://drive.google.com/file/d/13ra-bNFm-0AlwJXAZuBmZbdoqMy7fEI7/view?usp=drivesdk", 	
    "RoadAccidents/123" : "https://drive.google.com/file/d/13oAuHIkoBs56Ffv5bTZTogZ_iiR2c5ud/view?usp=drivesdk", 	
    "RoadAccidents/122" : "https://drive.google.com/file/d/13pTzefKVz2axR_KNLjmzeGEDgrTiX6WA/view?usp=drivesdk", 	
    "RoadAccidents/121" : "https://drive.google.com/file/d/13kqwP2Qo2NMuV8GbKlcYYNU4W22sXdG0/view?usp=drivesdk", 	
    "RoadAccidents/022" : "https://drive.google.com/file/d/13moIWeEXQOWebVLFO6qcqd4nJxLA__WU/view?usp=drivesdk", 	
    "RoadAccidents/021" : "https://drive.google.com/file/d/13kRbYR5jNL6XHz5SzidqO4K4F_zR8xaI/view?usp=drivesdk", 	
    "RoadAccidents/020" : "https://drive.google.com/file/d/13fJbGEmAYKx2a9QxD5Ga1kzviAnqb6AL/view?usp=drivesdk", 	
    "RoadAccidents/019" : "https://drive.google.com/file/d/13YsXhOL3XoGOVHMLdwBhhYefggXv6q4C/view?usp=drivesdk", 	
    "RoadAccidents/017" : "https://drive.google.com/file/d/13W-qeIgtO_Lq0UqX3vddn4JATmgj5bS7/view?usp=drivesdk", 	
    "RoadAccidents/016" : "https://drive.google.com/file/d/13XKpBSkxBBdkdys6VYF6eCpRQ_2_X3v6/view?usp=drivesdk", 	
    "RoadAccidents/012" : "https://drive.google.com/file/d/13PvCeGTdO_cttaHQoyPJucSxwAFNBTlz/view?usp=drivesdk", 	
    "RoadAccidents/011" : "https://drive.google.com/file/d/138XWPCz2kdpFOa56_cm0LOEO0M9ufEo0/view?usp=drivesdk", 	
    "RoadAccidents/010" : "https://drive.google.com/file/d/13CWAgWfOObuo3jz2QbQSPGVszjIaDFBN/view?usp=drivesdk", 	
    "RoadAccidents/009" : "https://drive.google.com/file/d/135hH7rfa9x5nNbB20z_x5cv2oaWbk_k3/view?usp=drivesdk", 	
    "RoadAccidents/004" : "https://drive.google.com/file/d/12y6CdpuX_nRq-43klJwxeVbg_QhBghN7/view?usp=drivesdk", 	
    "RoadAccidents/002" : "https://drive.google.com/file/d/12uOYWF_enGrMlZHkCm4o7EsivugeEoK9/view?usp=drivesdk", 	
    "RoadAccidents/001" : "https://drive.google.com/file/d/12wPNAP-N_1l6tAYuu5v8t7dJ5uaWLrH-/view?usp=drivesdk", 	
    "Fighting/047" : "https://drive.google.com/file/d/12l6ONNB3kwGGxn0JNrWrNWvNl9GS0AxG/view?usp=drivesdk", 	
    "Fighting/042" : "https://drive.google.com/file/d/12icGppDWMNNdEWe54ikbv966cAPhdSW1/view?usp=drivesdk", 	
    "Fighting/033" : "https://drive.google.com/file/d/12f99jTsVvvK-YqxEWugHpxLE14TFLHiq/view?usp=drivesdk", 	
    "Fighting/018" : "https://drive.google.com/file/d/12fNPlh5jq9e8puPJqUn_dQIW8yYfjdJS/view?usp=drivesdk", 	
    "Fighting/003" : "https://drive.google.com/file/d/12WAz6XzJafIyM27kHngRlGVetSTzo3hg/view?usp=drivesdk", 	
    "Explosion/043" : "https://drive.google.com/file/d/12PvilocLr-etMNOBmuqJ1HGR2imwLRRt/view?usp=drivesdk", 	
    "Explosion/039" : "https://drive.google.com/file/d/12NiB9haRGH0gaDEMvaJ5d1uoz39qjBr7/view?usp=drivesdk", 	
    "Explosion/036" : "https://drive.google.com/file/d/12HsEVrikbwdDqZIMPjbVM8zfta467rUw/view?usp=drivesdk", 	
    "Explosion/035" : "https://drive.google.com/file/d/12GwZXb6111qf64XdC4EofeV2Y3LmObKM/view?usp=drivesdk", 	
    "Explosion/033" : "https://drive.google.com/file/d/12Btc9RPPJFFKiRrBZJ4FUK0vex17yWs7/view?usp=drivesdk", 	
    "Explosion/029" : "https://drive.google.com/file/d/1250pE2DzYX1ab2B0IxQMVSF4XPFBonHS/view?usp=drivesdk", 	
    "Explosion/028" : "https://drive.google.com/file/d/126gCDYrsXzfBWq_NJCH3EvF72cxJoGxB/view?usp=drivesdk", 	
    "Explosion/027" : "https://drive.google.com/file/d/126js2RXOQ0dT80o-Ou7XQEsOcu9WFvXz/view?usp=drivesdk", 	
    "Explosion/025" : "https://drive.google.com/file/d/11zhhlL5M53S15HQ-PePU8-BsBUhTavCz/view?usp=drivesdk", 	
    "Explosion/022" : "https://drive.google.com/file/d/121y5X_EtNTfcsy8YDSNfmjCgqMMyiXbx/view?usp=drivesdk", 	
    "Explosion/021" : "https://drive.google.com/file/d/123vRU_HnBW_Ps-HYBeErnDoScLGQpBIY/view?usp=drivesdk", 	
    "Explosion/020" : "https://drive.google.com/file/d/11wieTvYFd22_hU6X2VDWNhF9smgQiJEG/view?usp=drivesdk", 	
    "Explosion/017" : "https://drive.google.com/file/d/11mj3zynz3nvX7YvkUMiUrJSfDR5bBv1e/view?usp=drivesdk", 	
    "Explosion/016" : "https://drive.google.com/file/d/11pFlBQhyfkFA7gm75tua4Ha-Oc1RRGW_/view?usp=drivesdk", 	
    "Explosion/013" : "https://drive.google.com/file/d/11uzGTI3sTewcpsKr6hSBJ6_-QQvYHD2U/view?usp=drivesdk", 	
    "Explosion/011" : "https://drive.google.com/file/d/11jmjcl-2TtolY2yu_g559w2w4KApx7jP/view?usp=drivesdk", 	
    "Explosion/010" : "https://drive.google.com/file/d/11lCSse5Bk4b9xgj9ZgURo4t7MNBOypF0/view?usp=drivesdk", 	
    "Explosion/008" : "https://drive.google.com/file/d/11gIwUPjEsrm_IeUB2Cxl78aLwN-tPyEF/view?usp=drivesdk", 	
    "Explosion/007" : "https://drive.google.com/file/d/11egOr43a_Cm7r-kd1GJRV78_OupJWAEI/view?usp=drivesdk", 	
    "Explosion/004" : "https://drive.google.com/file/d/11VDtxDE76U7OAtPY2YTQ0IQgJc85xIVD/view?usp=drivesdk", 	
    "Explosion/002" : "https://drive.google.com/file/d/11Atcpw9rvaTo0RpMv_b6U5zhkoWrK4_6/view?usp=drivesdk", 	
    "Burglary/092" : "https://drive.google.com/file/d/11P_vMBLw6Y7vAuV5srSf_hjpqvllo4KF/view?usp=drivesdk", 	
    "Burglary/079" : "https://drive.google.com/file/d/118trVHjr3qcH_gWQPG0V_lbbYXMQt8sK/view?usp=drivesdk", 	
    "Burglary/076" : "https://drive.google.com/file/d/117B7kEr74NGF2tnQWW0PNJIQkgHd4x2E/view?usp=drivesdk", 	
    "Burglary/061" : "https://drive.google.com/file/d/10tEkmdui0KlJARqwo1nceruOVM5HrsYj/view?usp=drivesdk", 	
    "Burglary/037" : "https://drive.google.com/file/d/11103mcmHFvLdKNViCHVOHP5sfLInSW9N/view?usp=drivesdk", 	
    "Burglary/035" : "https://drive.google.com/file/d/10fys43X5vs1B8L42DmU-9k1VsuC2bTza/view?usp=drivesdk", 	
    "Burglary/033" : "https://drive.google.com/file/d/10i5EFXD8CI-jX7wlpx30fNyj8zFsOw9N/view?usp=drivesdk", 	
    "Burglary/032" : "https://drive.google.com/file/d/10edFWCn-skNDEtrkV_i3dmLQSlaNHh1d/view?usp=drivesdk", 	
    "Burglary/024" : "https://drive.google.com/file/d/10b-k-GW_RwOQN5rHVvWuqLQek_hcztqJ/view?usp=drivesdk", 	
    "Burglary/021" : "https://drive.google.com/file/d/10apwMBVBn7pAlgkz9vDxvdIuJd1v0J3U/view?usp=drivesdk", 	
    "Burglary/018" : "https://drive.google.com/file/d/10Y8sEb665LOecBpxyqDBpQ1r6-5RQCxp/view?usp=drivesdk", 	
    "Burglary/017" : "https://drive.google.com/file/d/10YynTofEAaMoN420WA5JVfTeBHk40KG3/view?usp=drivesdk", 	
    "Burglary/005" : "https://drive.google.com/file/d/10WJQVkyDrtKJE7Qa1n2HvI8Yhe7fI6rW/view?usp=drivesdk", 	
    "Assault/011" : "https://drive.google.com/file/d/10TwdVXZW5Q-oPmXYnykAWFr7TnhWSn2M/view?usp=drivesdk", 	
    "Assault/010" : "https://drive.google.com/file/d/10RYTZH2nXZfw933Tatocn9-yVLa0Rh17/view?usp=drivesdk", 	
    "Assault/006" : "https://drive.google.com/file/d/10HWg6isL2DDkhtJ0cGB3KwaCIktokD7a/view?usp=drivesdk", 	
    "Arson/041" : "https://drive.google.com/file/d/109_m9vYC9sAL6LZNbky1l3apK_M20wjo/view?usp=drivesdk", 	
    "Arson/035" : "https://drive.google.com/file/d/10H4E5LMgijmcn4NWWRrxt2ErgrNbpSn3/view?usp=drivesdk", 	
    "Arson/022" : "https://drive.google.com/file/d/106rKw8WYUzvX-X74Z5_QMPvrLg6UKKLw/view?usp=drivesdk", 	
    "Arson/018" : "https://drive.google.com/file/d/102JzJupDmdc3-3dOaAu5iDYh5oWN0W8P/view?usp=drivesdk", 	
    "Arson/016" : "https://drive.google.com/file/d/103kNXccBECt4mvLqayKyNpMykwiISXLa/view?usp=drivesdk", 	
    "Arson/011" : "https://drive.google.com/file/d/1-krp2VUEsYm9HVC2kLx2PyHoQx4XFS3o/view?usp=drivesdk", 	
    "Arson/010" : "https://drive.google.com/file/d/1002G3mNtWcLd1Xbli_sDoo0UOEKCYdb_/view?usp=drivesdk", 	
    "Arson/009" : "https://drive.google.com/file/d/101dbqN2-Fw7twcVANibH2HuIvl4MBb78/view?usp=drivesdk", 	
    "Arson/007" : "https://drive.google.com/file/d/1-jRfbAxpSmAd-RQ8biXCxJupCobVF24y/view?usp=drivesdk", 	
    "Arrest/039" : "https://drive.google.com/file/d/1-gaLp3sKWGl9Ev5JasOLESFTVkiHq8pt/view?usp=drivesdk", 	
    "Arrest/030" : "https://drive.google.com/file/d/1-cPEpJt-UyeL-WdXwN0TpB-3pjJ8lW4-/view?usp=drivesdk", 	
    "Arrest/024" : "https://drive.google.com/file/d/1-Q8RNwfNEknRxfHRPxHGsYfYPDfro5YO/view?usp=drivesdk", 	
    "Arrest/007" : "https://drive.google.com/file/d/1-AV22JGFYsOWW6voTiwOqDjf24JdKA4-/view?usp=drivesdk", 	
    "Arrest/001" : "https://drive.google.com/file/d/1-Hf_Cf8Lq1fijcuauF2vV3LsVbS1GDoZ/view?usp=drivesdk", 	
    "Abuse/030" : "https://drive.google.com/file/d/1-4aOknNPApd1Du2S22B6r8mvaSlcQJjx/view?usp=drivesdk", 	
    "Abuse/028" : "https://drive.google.com/file/d/1-90CjYlAaBtU-UhQdoPId1JL0ZdhRfYq/view?usp=drivesdk",
    "Normal/935" : "https://drive.google.com/file/d/15V__jlJY9XffKL1CZ9m5_W5fwqj9ws--/view?usp=drivesdk",	
    "Normal/056" : "https://drive.google.com/file/d/1OJhx61ET4qo6cYtrgVLHThxhkNWAUvka/view?usp=drivesdk",	
    "Normal/365" : "https://drive.google.com/file/d/1ZW0GWeIokM1lK1Z1IAI1AlFgnidghLuO/view?usp=drivesdk",	
    "Normal/070" : "https://drive.google.com/file/d/12XZ_T8v_HPoRJENE_wXKqSq2RAGs9OEf/view?usp=drivesdk",	
    "Normal/901" : "https://drive.google.com/file/d/17-HWKnX5TWbl0oJHtvzBsYcUDMPrtcye/view?usp=drivesdk",	
    "Normal/881" : "https://drive.google.com/file/d/19oQCrLpcDBXAfFvTJd_UeUSafQzREl05/view?usp=drivesdk",	
    "Normal/051" : "https://drive.google.com/file/d/1AmTJVzvENeRZ-kfqSSpA6ksXM_qrzWIR/view?usp=drivesdk",	
    "Normal/634" : "https://drive.google.com/file/d/1BP0efUqWbZL0u4ZsGJmcD6bhBZN96iow/view?usp=drivesdk",	
    "Normal/603" : "https://drive.google.com/file/d/1HKtN0_n1oTMohBo5cXhgO2ndjIwe4Joo/view?usp=drivesdk",	
    "Normal/875" : "https://drive.google.com/file/d/1IYpL-ZSLZW0LpfpLVQlbcgM0kkeS_oCJ/view?usp=drivesdk",	
    "Normal/310" : "https://drive.google.com/file/d/1IyGQ9T2cAvPaf7En1Z3p2zmL8r4_WqCx/view?usp=drivesdk",	
    "Normal/018" : "https://drive.google.com/file/d/1KVaPiXxcT3KTUSWxORAVdS_yixKC9qwM/view?usp=drivesdk",	
    "Normal/189" : "https://drive.google.com/file/d/1MZ_EhMRtQgsfQn4Q2Nl7KI8dmgPQFdUx/view?usp=drivesdk",	
    "Normal/874" : "https://drive.google.com/file/d/1OasMiv1-gViJ9wtrKti9xLLzhDORchL2/view?usp=drivesdk",	
    "Normal/924" : "https://drive.google.com/file/d/1VlFg4W82N6JsV73IJEZQ-AA3jdWkfim2/view?usp=drivesdk",	
    "Normal/210" : "https://drive.google.com/file/d/1WoKE8NFXuolj3Jv5rp2Ip0B9bH_jTT3_/view?usp=drivesdk",	
    "Normal/251" : "https://drive.google.com/file/d/1aMDZYjbPbVOfZUMS77r04k-tEyKMiPNT/view?usp=drivesdk",	
    "Normal/898" : "https://drive.google.com/file/d/1aaHoyCUY-H7H5uVHfhnJHYfQoxfbzQIP/view?usp=drivesdk",	
    "Normal/892" : "https://drive.google.com/file/d/1bkzm5pzMRg--TE14s95TMoc7vovQG263/view?usp=drivesdk",	
    "Normal/910" : "https://drive.google.com/file/d/1d2WBdN62TnrCHvyoPmz9S_brTtC2jdo0/view?usp=drivesdk",	
    "Normal/067" : "https://drive.google.com/file/d/13KEFOzBmEZqnrXemflbT4TR2g5fiuIpX/view?usp=drivesdk",	
    "Normal/656" : "https://drive.google.com/file/d/13xk39nQIiQi26iBNtmvLY0sB7qzBQ6fF/view?usp=drivesdk",	
    "Normal/780" : "https://drive.google.com/file/d/149n8SgnfNqjC8FSlA95ngr_O5zKafSHC/view?usp=drivesdk",	
    "Normal/641" : "https://drive.google.com/file/d/14cjsbQ7_9N64QckNjKL_S0iH0EqI-Q6D/view?usp=drivesdk",	
    "Normal/906" : "https://drive.google.com/file/d/15bgFm0H96sFkIkG1BOMtUdb2kXHf02uN/view?usp=drivesdk",	
    "Normal/417" : "https://drive.google.com/file/d/15xPfCFhpnVV_WC1Y6I-oWV5Q54jyEfve/view?usp=drivesdk",	
    "Normal/360" : "https://drive.google.com/file/d/16kFvxJTeOFnKyDQ5xenFKN2yYBVvWCjz/view?usp=drivesdk",	
    "Normal/876" : "https://drive.google.com/file/d/180n03Ioaqkwl_lNixLS17mlNfAyjde_6/view?usp=drivesdk",	
    "Normal/888" : "https://drive.google.com/file/d/18RCpyJIDIkLJIARjRcWEtXpzBVNg3STV/view?usp=drivesdk",	
    "Normal/887" : "https://drive.google.com/file/d/1AZhqsRuLoQ9YgznFshPPJUfax8mr15uD/view?usp=drivesdk",	
    "Normal/175" : "https://drive.google.com/file/d/1BAclsNArt2eLYH_cnQfxiKa5menFZnV5/view?usp=drivesdk",	
    "Normal/702" : "https://drive.google.com/file/d/1BkH-kdUl2JB71WLzVx_FvKgstYLVaDxF/view?usp=drivesdk",	
    "Normal/884" : "https://drive.google.com/file/d/1CYwZbVnBYW1y8GR3rhzfzuNBIGQpcy_v/view?usp=drivesdk",	
    "Normal/725" : "https://drive.google.com/file/d/1EkIeow8yny5pvKCWdsl1zTl8RII5DlAm/view?usp=drivesdk",	
    "Normal/828" : "https://drive.google.com/file/d/1HRECsacV-bP3CVYWprrz2A-Q6kv5qXJY/view?usp=drivesdk",	
    "Normal/938" : "https://drive.google.com/file/d/1Is9mT6sFTRhnApXy3GyV5vEVK0AVp3uA/view?usp=drivesdk",	
    "Normal/778" : "https://drive.google.com/file/d/1M6flKhFoCAkxk-avFN2rdmpSjwUGi5TL/view?usp=drivesdk",	
    "Normal/871" : "https://drive.google.com/file/d/1MejaKvbZezYWwvFZzY3tyMKx3_bKmEYd/view?usp=drivesdk",	
    "Normal/900" : "https://drive.google.com/file/d/1NWEfpKuEb7ElMMv3ixZxzV7-XCf0cST-/view?usp=drivesdk",	
    "Normal/866" : "https://drive.google.com/file/d/1P_Hy7_cY279EbmAxpS_WT3Bx6zneQhpi/view?usp=drivesdk",	
    "Normal/943" : "https://drive.google.com/file/d/1T0EdKeSE9ZRrcuB3Klq-h8BOA6oFP3zc/view?usp=drivesdk",	
    "Normal/915" : "https://drive.google.com/file/d/1U9b-E9lVhbnG2UoPjnHn-h6850IEPlgM/view?usp=drivesdk",	
    "Normal/439" : "https://drive.google.com/file/d/1UAzGk2WFqT7NceO697eL_tyulnKDAJLX/view?usp=drivesdk",	
    "Normal/913" : "https://drive.google.com/file/d/1dcd1xy2_tPIpJpQa60VKzCwBNQLnRdsK/view?usp=drivesdk",	
    "Normal/798" : "https://drive.google.com/file/d/1dhNXSst99QyUC3U3feN9vFqaBabvFRRQ/view?usp=drivesdk",	
    "Normal/896" : "https://drive.google.com/file/d/1fumG88l7O5P0QSPuoZd0xtJZGGFbtTsI/view?usp=drivesdk",	
    "Normal/696" : "https://drive.google.com/file/d/1gBhzCsPeB1D-QgjYOOw0k-r4LvHBzy2n/view?usp=drivesdk",	
    "Normal/050" : "https://drive.google.com/file/d/1iZNk2C-K4cO2Tg1STXC9BCN9EknXj8ZX/view?usp=drivesdk",	
    "Normal/027" : "https://drive.google.com/file/d/1iiDreF4LwGHHvpZEYU6-eS93V-4R8ZAA/view?usp=drivesdk",	
    "Normal/758" : "https://drive.google.com/file/d/1jZS8dnLS6k-so9GCVK1wlN2hceKKXwUR/view?usp=drivesdk",	
    "Normal/352" : "https://drive.google.com/file/d/1nG2ooy9dvOwteX7beUQ6LwFE-pbDlN0R/view?usp=drivesdk",	
    "Normal/129" : "https://drive.google.com/file/d/1ooIZzlCw9-dU0jasYxSlcJLZdZwfMdTM/view?usp=drivesdk",	
    "Normal/873" : "https://drive.google.com/file/d/1p6Hbyz9cCs3E7XGwXFoKkfACwBdZvr1P/view?usp=drivesdk",	
    "Normal/878" : "https://drive.google.com/file/d/1qKgawvytQM4GF2oZTao5SXHXBGYza6e6/view?usp=drivesdk",	
    "Normal/877" : "https://drive.google.com/file/d/1rr5UJ9IFTvV0-CBCRDg4cclYFJdCf4gS/view?usp=drivesdk",	
    "Normal/894" : "https://drive.google.com/file/d/1u4leB4tPgWY4Zy4EudFKdSVeInOXwjSG/view?usp=drivesdk",	
    "Normal/867" : "https://drive.google.com/file/d/1vCZdFXZJmMqOWNDB6iD4LHIgeod8MJZ9/view?usp=drivesdk",	
    "Normal/781" : "https://drive.google.com/file/d/1vJfI1E3ozZLzjmMB7kTSSHh4YOueQ8nW/view?usp=drivesdk",	
    "Normal/150" : "https://drive.google.com/file/d/1yZoKzewDB_tNsjPFeW3emOM5erv2m4Kx/view?usp=drivesdk",	
    "Normal/885" : "https://drive.google.com/file/d/1zyi5DTlb9LjSmZT1jGFhWAvTnjqg6Wzm/view?usp=drivesdk",	
    "Normal/782" : "https://drive.google.com/file/d/11f3J1ob3iQiQp0m4DPDXaBaJ4HUc2Gf1/view?usp=drivesdk",	
    "Normal/936" : "https://drive.google.com/file/d/11gPUbvm2JcBYlf8x6a_fJ29mjcCMtZt6/view?usp=drivesdk",	
    "Normal/880" : "https://drive.google.com/file/d/11q1HP-ueqyFIXm6exIW2_UkK-HTTXb-T/view?usp=drivesdk",	
    "Normal/944" : "https://drive.google.com/file/d/1339G66uwSs6iGK2Z4RY7dqoWAetolLni/view?usp=drivesdk",	
    "Normal/041" : "https://drive.google.com/file/d/13wGLE6wWqJaagRnncbNaza_T8bblwx-x/view?usp=drivesdk",	
    "Normal/196" : "https://drive.google.com/file/d/15B0PVexLOtDDFYlR1aVcDozjXC43-3Zz/view?usp=drivesdk",	
    "Normal/801" : "https://drive.google.com/file/d/15m-dMkMxw3arB8Ub02JaEr0ClFozSLhn/view?usp=drivesdk",	
    "Normal/224" : "https://drive.google.com/file/d/1BxSkkgulxOO3xl9JHpaxysAP41SKFh5r/view?usp=drivesdk",	
    "Normal/889" : "https://drive.google.com/file/d/1CpEs1j11OR4P75N67rg3LtE6Hw-5oRqO/view?usp=drivesdk",	
    "Normal/926" : "https://drive.google.com/file/d/1EbxHnQ1MYM0hA3Vt9-PAa-87v2JXizDE/view?usp=drivesdk",	
    "Normal/905" : "https://drive.google.com/file/d/1EcYXTY4xa9xHtY_mrAIaRtjMxrJ35RK0/view?usp=drivesdk",	
    "Normal/925" : "https://drive.google.com/file/d/1EjJPCHGR0pS2FwrE9t1F5KQ0GK58HqK5/view?usp=drivesdk",	
    "Normal/317" : "https://drive.google.com/file/d/1F08m1HFcoHgh0v-BurT-BFhga1NiAQWq/view?usp=drivesdk",	
    "Normal/247" : "https://drive.google.com/file/d/1Htp33HCIzmVKrvuLgb5F3wmTKd1FgohI/view?usp=drivesdk",	
    "Normal/783" : "https://drive.google.com/file/d/1L9MiN1Ij8Z7QEFvL5YodviChPuXNqzQm/view?usp=drivesdk",	
    "Normal/345" : "https://drive.google.com/file/d/1LQG-W5tBA5IQGWSfkryROwwpHGq78bcn/view?usp=drivesdk",	
    "Normal/891" : "https://drive.google.com/file/d/1NiwzX-D45yS4kVpURg_YAXlTdIBmip_f/view?usp=drivesdk",	
    "Normal/904" : "https://drive.google.com/file/d/1PKg-Dq8WwB2acLvnv3w8sluuYy7JxPfA/view?usp=drivesdk",	
    "Normal/006" : "https://drive.google.com/file/d/1Rohvr88TorafzWts1BRmr_r7qoh4yxS6/view?usp=drivesdk",	
    "Normal/931" : "https://drive.google.com/file/d/1TtIzpDG0R169w_VujHD6QxJiQSnE3DA7/view?usp=drivesdk",	
    "Normal/722" : "https://drive.google.com/file/d/1UO22wiMnpvBS7vEXQ7WOk6UCr065JlQ_/view?usp=drivesdk",	
    "Normal/010" : "https://drive.google.com/file/d/1VfuzVZdyI7Hw-O2n1t1XKgP3YpStzG0O/view?usp=drivesdk",	
    "Normal/015" : "https://drive.google.com/file/d/1ZRlAuCIlxAPHZjYiT2H5dFPd5a-YGDtK/view?usp=drivesdk",	
    "Normal/059" : "https://drive.google.com/file/d/1aXAcxy3gfil-4rdoyHovsfFw8kjcg4sm/view?usp=drivesdk",	
    "Normal/886" : "https://drive.google.com/file/d/1avSowesQJ3S3LUeMDGUHyVC1XTHgMoJN/view?usp=drivesdk",	
    "Normal/903" : "https://drive.google.com/file/d/1bQxgpvewd5_2JJJYtrUcHgvKz_OiDVj7/view?usp=drivesdk",	
    "Normal/246" : "https://drive.google.com/file/d/1dtFqCDpmoTiT92QTA6qIZm03ne7wWrjU/view?usp=drivesdk",	
    "Normal/014" : "https://drive.google.com/file/d/1fj2SK_3TZYE-g9eMPI1parpjgDK6c9J-/view?usp=drivesdk",	
    "Normal/941" : "https://drive.google.com/file/d/1gU5JQlYARa4G0NfSdFogFTOwE5LZQp33/view?usp=drivesdk",	
    "Normal/908" : "https://drive.google.com/file/d/1iAMP4EFv0C63AGo71jDtXdV9alOX3LAJ/view?usp=drivesdk",	
    "Normal/914" : "https://drive.google.com/file/d/1k1ecVVjQflk743lRXIS8wcOkFIRQetrS/view?usp=drivesdk",	
    "Normal/024" : "https://drive.google.com/file/d/1kx0cQPzm0d98dbj0pehHiukurFGdHua4/view?usp=drivesdk",	
    "Normal/872" : "https://drive.google.com/file/d/1newiDJpNcg4y60sEUPlobC2Uma4O7T55/view?usp=drivesdk",	
    "Normal/897" : "https://drive.google.com/file/d/1o03EaEzAjRlIvx_nYIr0MExEJbh7JCli/view?usp=drivesdk",	
    "Normal/182" : "https://drive.google.com/file/d/1opxTWQb3W5-xbpmIimU0G6oYYpQUyC3Y/view?usp=drivesdk",	
    "Normal/168" : "https://drive.google.com/file/d/1rV9ozx2iHkR2FGEozdeSFwXyDzb-8qLk/view?usp=drivesdk",	
    "Normal/019" : "https://drive.google.com/file/d/1tg03kkEX4ZfHtJCrdfqYkPK5Rovr5nxx/view?usp=drivesdk",	
    "Normal/879" : "https://drive.google.com/file/d/1u602-Rqvad-lXxS9jIHhIsu4HPk30Vu2/view?usp=drivesdk",	
    "Normal/882" : "https://drive.google.com/file/d/1vHix8bTuoXxdWYstQH_vzCf4n0yKkvwS/view?usp=drivesdk",	
    "Normal/890" : "https://drive.google.com/file/d/1y7QPo014E3vwwT83L24DJXyTo-BaWz4D/view?usp=drivesdk",	
    "Normal/911" : "https://drive.google.com/file/d/1-OMIXaqcmldSp4KY38hpjUmK3HZZ4hn_/view?usp=drivesdk",	
    "Normal/033" : "https://drive.google.com/file/d/1Uqj7e37YLczu3nRpRC3CvyvfluiAn6Uu/view?usp=drivesdk",	
    "Normal/927" : "https://drive.google.com/file/d/1-RPQvCA-dxbWldhr5WWRXNtkAANzpKLo/view?usp=drivesdk",	
    "Normal/704" : "https://drive.google.com/file/d/10Nn9ZzleSo89XrgfaSapyHkV0CinS0gd/view?usp=drivesdk",	
    "Normal/034" : "https://drive.google.com/file/d/10UlTpf6lEXkhKNqnbN0r7xDoGGBx0Ne9/view?usp=drivesdk",	
    "Normal/289" : "https://drive.google.com/file/d/152-AuNhoy9ic3vojSn-iU5qmkcgcpajK/view?usp=drivesdk",	
    "Normal/686" : "https://drive.google.com/file/d/1953Znd82gZRTZ5V1mbMUN_cly3bFLsbG/view?usp=drivesdk",	
    "Normal/940" : "https://drive.google.com/file/d/1AF4B0bT3xp3pKFafxDjyErD1RJI5O9f0/view?usp=drivesdk",	
    "Normal/930" : "https://drive.google.com/file/d/1CMgTAJWZ3s4-1RTeeOQCqQ1SqTppdfB0/view?usp=drivesdk",	
    "Normal/203" : "https://drive.google.com/file/d/1D4K0lHAyrZ7k_mEukJIC3nQhu9oDf_dx/view?usp=drivesdk",	
    "Normal/928" : "https://drive.google.com/file/d/1FX1pLgaPyRx7XQfWw7qDI5j5esIrdD7l/view?usp=drivesdk",	
    "Normal/606" : "https://drive.google.com/file/d/1Ib_v9XWcj6XK1UrpbQhudVsL1eu6bQxV/view?usp=drivesdk",	
    "Normal/831" : "https://drive.google.com/file/d/1JYFNJp0ZJTyP7vTJZUHHSeh5DNxPGDYH/view?usp=drivesdk",	
    "Normal/937" : "https://drive.google.com/file/d/1KxYlYh7RiKpVdDo8Dr28MP6TI1FtIqbo/view?usp=drivesdk",	
    "Normal/907" : "https://drive.google.com/file/d/1L91X6mnnd9zIYCj48JBOntTELEGF7g_x/view?usp=drivesdk",	
    "Normal/048" : "https://drive.google.com/file/d/1MgIn77CkBBpq5jv5YQTy6hunP_ZqbxGa/view?usp=drivesdk",	
    "Normal/899" : "https://drive.google.com/file/d/1OHMhzLYu6xTkXxhgYB4vufgSqCMzlua8/view?usp=drivesdk",	
    "Normal/939" : "https://drive.google.com/file/d/1PW9KF9O9vsGkTyiSe5M4Tt_a_dQYrJom/view?usp=drivesdk",	
    "Normal/745" : "https://drive.google.com/file/d/1Q01xqf_Nv209EKKS67RcBACtD6_pn8mk/view?usp=drivesdk",	
    "Normal/929" : "https://drive.google.com/file/d/1SbbQmm4OOsrm2p_OOxZHji3HvUCvdXKr/view?usp=drivesdk",	
    "Normal/003" : "https://drive.google.com/file/d/1USxGhGXvuHF7HIMV0BA217pNaB2CvhX_/view?usp=drivesdk",	
    "Normal/217" : "https://drive.google.com/file/d/1UYslyjzKKakbuPKnR_TFq0k60B2AgH6q/view?usp=drivesdk",	
    "Normal/597" : "https://drive.google.com/file/d/1WvAw_bPcPFOXATSxYGJUFwW8K8PRcSbm/view?usp=drivesdk",	
    "Normal/576" : "https://drive.google.com/file/d/1XBG-Xw-k20ewJGMGygz_NHboIO8Rzr6D/view?usp=drivesdk",	
    "Normal/478" : "https://drive.google.com/file/d/1XtHztKBr0LUHLxHBImj3Xhw4mobEqfA9/view?usp=drivesdk",	
    "Normal/100" : "https://drive.google.com/file/d/1_KAMrDAlaxLydjTcS9ung1O4kx0hUYAF/view?usp=drivesdk",	
    "Normal/909" : "https://drive.google.com/file/d/1_vmrlTVd_NFo81rYcW1r5TyW14Ouytnl/view?usp=drivesdk",	
    "Normal/025" : "https://drive.google.com/file/d/1aYQpIBJnSdrWLZrbjVa4hPFDs-nfqefd/view?usp=drivesdk",	
    "Normal/902" : "https://drive.google.com/file/d/1cw4b8LKkTh5QTI4Qk6zQyixfmHhF29UP/view?usp=drivesdk",	
    "Normal/063" : "https://drive.google.com/file/d/1f6__HCx0n-GV1_5jd6s6LtVWbBokllHR/view?usp=drivesdk",	
    "Normal/717" : "https://drive.google.com/file/d/1fpnvwvR8MRztfi1-0NYKzGCmlZSYJNeL/view?usp=drivesdk",	
    "Normal/895" : "https://drive.google.com/file/d/1hSeCVH7GvafXiY6bakKMeNL3BKV2KFqS/view?usp=drivesdk",	
    "Normal/923" : "https://drive.google.com/file/d/1iSVGTxeWpAuQB0r01n7Z75xJb2005e19/view?usp=drivesdk",	
    "Normal/883" : "https://drive.google.com/file/d/1nO9_OW1FyOGajnTR8cDlZ3w4e_MdKn2S/view?usp=drivesdk",	
    "Normal/869" : "https://drive.google.com/file/d/1n_gcsKN3f2LPgxYxLEO7Db4-uQriV0Za/view?usp=drivesdk",	
    "Normal/710" : "https://drive.google.com/file/d/1qEGX-YwFQhc4djlGZ4qu6EghvtWCuRc1/view?usp=drivesdk",	
    "Normal/932" : "https://drive.google.com/file/d/1s9MEVzuGfVF2wIFs2VhCMeh1RCo-EmQK/view?usp=drivesdk",	
    "Normal/870" : "https://drive.google.com/file/d/1sTd8thaxX8ZMz1m4D_4FSNuH4_fcWo_B/view?usp=drivesdk",	
    "Normal/868" : "https://drive.google.com/file/d/1uE5XnErGHooaZHLh1wPzphCcL7qGwplo/view?usp=drivesdk",	
    "Normal/621" : "https://drive.google.com/file/d/1yzJVxf1mJSVE1vEQKi5wqg0SLKn5nfdz/view?usp=drivesdk",	
    "Normal/312" : "https://drive.google.com/file/d/12etnb_tpnMMUYUwF2fwI5R_iuSFPZNum/view?usp=drivesdk",	
    "Normal/934" : "https://drive.google.com/file/d/13PQViwmd24uhpmo2UdPT7uiZuvUpR6nu/view?usp=drivesdk",	
    "Normal/912" : "https://drive.google.com/file/d/1CG-RB1_SFvJlv9XFDgISj6OyQmsSuOr6/view?usp=drivesdk",	
    "Normal/893" : "https://drive.google.com/file/d/1FT715mOeQ-EadBFXQiCtE8eonmzCLFnj/view?usp=drivesdk",	
    "Normal/453" : "https://drive.google.com/file/d/1Q-SjWiIriVcdY_bv8k6i-w3tfyz1Hy53/view?usp=drivesdk",	
    "Normal/042" : "https://drive.google.com/file/d/1QJ2W4hJvObfOQKeWgr-4vF7toUVV68p9/view?usp=drivesdk",	
    "Normal/933" : "https://drive.google.com/file/d/1S0R1BtL5p5pMD-dqXqk44m3nVY_CCCAF/view?usp=drivesdk",	
    "Normal/401" : "https://drive.google.com/file/d/1Zcyfpa1JSCVZy8IdeZqN0z6w7ziBTlrd/view?usp=drivesdk",	
    "Normal/452" : "https://drive.google.com/file/d/1lIRVdqqvJ0oABLekPFMtc-RV9y0ZHSjo/view?usp=drivesdk",	
    "Normal/248" : "https://drive.google.com/file/d/1qO0LAom6UJGkg-CWA8BTqQ8IQK_d6jbr/view?usp=drivesdk",	
}
# experiment = ["RoadAccidents/002", "Explosion/008"]

def read_anno_file(f):
    # anno = np.genfromtxt(f, dtype={'names': ('class', 'video', 's1', 'e1', 's2', 'e2'),
    #    'formats': (dtype=str, np.int, np.int, np.int, np.int, np.int)})\
    anno = np.genfromtxt(f, dtype=str, comments="{% comment %}")
    # fw = open(os.path.join(root, "annotations", "video_refs.txt"), "w")
    # for i in range(anno.shape[0]):
        # a = anno[i]
        # tplt = "gdrive/MyDrive/honours_project/c3d_features/data/anomaly-videos/{}/{}{}_x264.mp4".format(a[0], a[0], a[1])
        # fw.write("{}\n".format(tplt))
    # fw.close()
    return anno


def verify_anom_fc6(scores, n_frame, anom_window, threshold, prev_res):
    n_scores = 32
    start_anom = int(anom_window[0])
    end_anom = int(anom_window[1])
    start_anom = max(0, int((start_anom/n_frame)*n_scores - 4))
    end_anom = min(n_scores, ceil((end_anom/n_frame)*n_scores + 4))
    anno_sgms = sorted(list(set([max(0,start_anom-1), end_anom-1])))

    # start_anom = int(((anom_window[0])/n_frame)*n_scores)
    # end_anom = int((min(anom_window[1]+buffer, n_frm)/n_frm)*n_scores)
    
    
    # print("n_frm:{}".format(n_frm))
    # print("n_scores:{}".format(n_scores))
    # print("start_anom_frm:{}".format(anom_window[0]))
    # print("end_anom_frm:{}".format(anom_window[1]))
    # print("start_anom_scores:{}".format(start_anom))
    # print("end_anom_scores:{}".format(end_anom))
    # print("anno_sgms:{}".format(anno_sgms))
    
    if len(anno_sgms) == 1: #TODO temp fix for overlapping small sgm annotation - may be root of error
        anno_sgms.append(anno_sgms[0]+1)

    for i in range(anno_sgms[0], anno_sgms[1]):
        if scores[i] >= threshold:
            
            return True, anno_sgms
    return (False or prev_res), anno_sgms
        

def frame_averaging(dataset):
    sims_d_div16 = {}
    for lbl, data in dataset.items():
        avgd_data  = []
        sz  = len(data)/16
        for i in range(1, int(sz)):
            if i == sz-1:
                avgd_data.append(np.average(data[(i-1)*16:]))
            else:
                avgd_data.append(np.average(data[(i-1)*16: i*16]))
        sims_d_div16[lbl] = avgd_data
    return sims_d_div16      

def individual_scale(values_d):
    scaled_values = {}
    for lbl, val in values_d.items():
        s = np.std(val)
        u = np.average(val)
        val_sc = [(x-u)/s for x in val]
        scaled_values[lbl] = val_sc
    return scaled_values
      
def filter_to_zero(values_d, t):
    """
    utility function which clamps values below a certain std dev to 0
    NB: expects a dictionary of lists which are standardized (x-u/s)
    """
    filtered_values = {}
    for lbl, val in values_d.items():
        # print("scaled vals")
        # print(len(val))
        # print(val)
        val_filt = list(map(lambda x: 0 if x < t else x, val))
        # print(len(val_filt))
        # assert len(val_filt) == 32, "filtering is losing values (not 32)"
        # print("filtered vals")
        # print(val_filt)
        filtered_values[lbl] = val_filt
    return filtered_values


def relative_scale(sims, group, cutoff):
    scaled_sims = {}
    batch_values = []
    count = 0
    lbls = list(sims.keys())
    shuffle(lbls)
    
    for lbl in lbls:
        sim = sims[lbl]
        batch_values.extend(sim)
        count +=1
        if count % group == 0 or count == len(sims):
            if count % group == 0:
                subset = islice(sims.items(), count-group, count)
            else:
                back_count = 0
                while (int(count-back_count) % group != 0):
                    back_count+=1
                subset = islice(sims.items(), count-back_count, count)
                
                
            s = np.std(batch_values)
            u = np.average(batch_values)
            m = np.max(batch_values)
            ms = (m-u)/s
            
            for lb, si in subset:
                sim_sc = [(x-u)/s for x in si]
            # print("initial sim len: ", len(sim))
                # mms = MinMaxScaler()
                # scaled = mms.fit_transform(clipped.reshape(1,-1))
                sim_sc = np.clip(np.asarray(sim_sc), cutoff, np.max(sim_sc)) #atleast clip anom scores that are less than 1 std devs from avg of batch
                # mx = max(sim_sc)
                # mn = min(sim_sc)
                # sim_sc_norm = [(x-mn)/(mx-mn) for x in sim_sc]
                scaled_sims[lb] = sim_sc
                
            batch_values = []
            
    return scaled_sims
            
def to_range(values_d, r):
    range_d = {}
    for lbl, values in values_d.items():
        mx = float(max(values))
        mn = float(min(values))
        if int(mx) == 0:
            values_norm = [0 for x in values]
        else:
            values_norm = [float(r*(x-mn)/(mx-mn)) for x in values]
        range_d[lbl] = values_norm
    return range_d

def bulkrangeto1(sims, group, cutoff):
    scaled_sims = {}
    batch_values = []
    count = 0
    lbls = list(sims.keys())
    shuffle(lbls)
    
    for lbl in lbls:
        sim = sims[lbl]
        batch_values.extend(sim)
        count +=1
        if count % group == 0 or count == len(sims):
            if count % group == 0:
                subset = islice(sims.items(), count-group, count)
            else:
                back_count = 0
                while (int(count-back_count) % group != 0):
                    back_count+=1
                subset = islice(sims.items(), count-back_count, count)
                
                
            mx = np.max(batch_values)
            mn = np.min(batch_values)
            # print("min", mn)
            # print("max", mx)
            for lb, si in subset:
            # print("initial sim len: ", len(sim))
                # mms = MinMaxScaler()
                # scaled = mms.fit_transform(clipped.reshape(1,-1))
                sim_sc = [(x-mn)/(mx-mn) for x in si]
                # sim_sc = np.clip(np.asarray(sim_sc), cutoff, max(sim_sc)) #atleast clip anom scores that are less than 1 std devs from avg of batch
                scaled_sims[lb] = sim_sc
                
            batch_values = []
            
    return scaled_sims
    
def peak_window(scores_d, wdw):
    transform_scores_d = {}
    # print("[PEAK WDW] window size: {}".format(wdw))
    for lbl, scores in scores_d.items():
        transform_scores = []
        for f in range(0, len(scores)):
            # print("[PEAK WDW] frame: ", f)
            if f > wdw and (len(scores)-f-1) > wdw:

                before = sum(scores[f-wdw:f])
                after = sum(scores[f:f+wdw])
                step = after - before
                # print("\t[PEAK WDW] valid f -> score is ", scores[f])
                # print("\t[PEAK WDW] valid f -> step is ", step)
                # print("\t[PEAK WDW] valid f -> transformed scores is ", scores[f]+step)
                # print()
                #TODO limit to 0 or allow negatives?
                transform_scores.append(max(scores[f], scores[f] + step))
            else:
                # print("\t[PEAK WDW] f not in valid wdw range")
                transform_scores.append(scores[f])
                
        transform_scores_d[lbl] = transform_scores
    return transform_scores_d

def delta_multiplier(sims_d, delta_d):
    mul_scores_d = {}
    for lbl, scores in sims_d.items():
        mul_scores = []
        deltas = delta_d[lbl]
        d_avg = np.average(deltas)
        for x, m in zip(scores, deltas):
            mul_scores.append((m/d_avg)*x)
        mul_scores_d[lbl] = mul_scores
    return mul_scores_d
    
def center_on_avg_err(sims_d):
    mul_scores_d = {}
    for lbl, scores in sims_d.items():
        mul_scores = []
        s_avg = np.average(scores)
        for x in scores:
            mul_scores.append(x-s_avg)
        mul_scores_d[lbl] = mul_scores
    return mul_scores_d

def convert_to_32(profile):
    profile_32 = {}
    for lbl, data in profile.items():
        new_data = []
        data = np.asarray(data)
        # print("convert these scores to 32:{}".format(data.shape))
        sgms = np.array_split(data, 32)
        for sgm in sgms:
            if len(sgm.tolist()) == 0:
                new_data.append(0)
            else:
                new_data.append(np.max(sgm))
        # print("len of new scores: {}".format(len(new_data)))
        profile_32[lbl] = new_data
    return profile_32

def decompose(lkkm_d):
    dcmp = {}
    for lbl, lkkm in lkkm_d.items():
        # print(lbl)
        # print(lkkm)
        if len(lkkm) > 10:
            score_frame = pd.DataFrame({
                # "index": np.arange(len(avgd_data)),
                "data": lkkm   
            })
            # score_frame.set_index('index',inplace=True)
            # score_frame.index=pd.to_datetime(score_frame.index)
            score_frame.fillna(0, inplace=True)
            result = seasonal_decompose(score_frame, model='additive', period=min(int(len(score_frame["data"])/2), PER))
            resid_frame = pd.DataFrame({
                "data": result.resid   
            })
            resid_frame.fillna(0, inplace=True)
            
            # print(type(result.resid))
            # rst = np.asarray(result.resid, dtype=np.float32)            
            # print(type(rst))
            # print(rst)
            dcmp[lbl] = np.asarray(resid_frame["data"], dtype=np.float32)
            # resid_frame = pd.DataFrame({
                # "data": result.resid   
            # })

            # resid_frame.dropna(inplace=True)
            # result2 = seasonal_decompose(resid_frame, model='additive', period=min(int(len(resid_frame["data"])/2), 20))
            # result2.plot()            

            # final = (result.trend + result2.resid)/2
            # final.plot()
            # plt.title("final {}".format(vid))
            # plt.show()
        
    return dcmp
#TODO



def nan_to_zero(values_d):
    filled_d = {}
    for lbl, val in values_d.items():
        df = pd.DataFrame({   
            "data": val,
        })
        df["data"] = df["data"].fillna(0)
        filled_d[lbl] = df["data"].values
        # print(df["data"].values)
    return filled_d

def broadc(vals, splt):
    broad_list = []
    sgms = np.array_split(vals, splt)
    for sgm in sgms:
        l = sgm.tolist()
        m = max(l)
        broad = [m]*int(32/splt)
        broad_list.extend(broad)
    assert len(broad_list) == 32
    # print(broad_list)
    return broad_list

def apply_broadc(values_d, splt):
    broad_d = {}
    for lbl, val in values_d.items():
        broad_d[lbl] = broadc(val, splt)
    return broad_d

def add_scores(vals1, vals2):
    combined = [x+y for x,y in zip(vals1, vals2)]
    return combined

def combine_heuristics(sims_d, lkkm_d):
    hrts_d = {}
    for lbl, lkkm in lkkm_d.items():
        craft = sims_d[lbl]
        lk = broadc(lkkm, 8)
        cr = broadc(craft, 8)
        combine = add_scores(cr, lk)
        # combine = np.add(lk, cr).tolist()
        hrts_d[lbl] = combine
    hrts_d["RoadAccidents/021"] = sims_d["RoadAccidents/021"]        
    return hrts_d
    
def smoothing(values_d, wdw):
    smooth_d = {}
    for lbl, val in values_d.items():
        c = []
        for i in range(len(val)):
            if i > wdw and (len(val)-i-1) > wdw:
                agg = sum(val[i-wdw:i+wdw])
                agg /= wdw
            elif i < wdw:
                agg = sum(val[:i])
                agg /= max(1,i)
            elif (len(val)-i-1) < wdw:
                agg = sum(val[i:])
                agg /= max(1,i)
            c.append(agg)
        smooth_d[lbl] = c
    return smooth_d

def score_combine(scores_d, sims_d, lkkm_d):
    
    #consider multiplying sims_d by avg_delta/delta_i, therefore we amplify for the smaller scores -> that would have to happen in relative_scale
    #SIMS
    # sims = relative_scale(sims, 150, 0.5)
    # sims = delta_multiplier(sims, deltas_d)
    # sims = peak_window(sims, 20)
    # sims = center_on_avg_err(sims)
    # sims = bulkrangeto1(sims, 2, 1)
    
    #MIL
    #LKKM
    # lkkm_d = decompose(lkkm_d)
    # lkkm_d = relative_scale(lkkm_d, 50, 0.5)
    # lkkm_d = peak_window(lkkm_d, 20)
    # lkkm_d = center_on_avg_err(lkkm_d)
    # lkkm_d = bulkrangeto1(lkkm_d, 2, 1)
    
    
    #COMBINED

    combine_d = {}
    #TODO rmember that scores_d and sims_d may be different sizes
    #TODO fix tmp -> ust use lbl when we have correct keys in annMIL data (predictions)
    for lbl, lkkm in lkkm_d.items():
        craft = sims_d[lbl]
        if len(lkkm) > len(craft):
            x = np.asarray(craft)
            x.resize(len(lkkm))
            y = np.asarray(lkkm, refcheck=False)
        else:
            x = np.asarray(lkkm)
            x.resize(len(craft), refcheck=False)
            y = np.asarray(craft)
        combine = np.add(x, y)
        # print(combine)
        combine_d[lbl] = combine.tolist()
    combine_d["RoadAccidents/021"] = sims_d["RoadAccidents/021"]        
    profile_d = consensus(combine_d, scores_d, wdw=6)
    
    # profile_d = center_on_avg_err(profile_d)
    # profile_d = individual_scale(profile_d, 1.4)
    # profile_d = rangeto1(profile_d)
        
    profile_d = center_on_avg_err(profile_d)
    profile_d = bulkrangeto1(profile_d, 3, 0)


    #peak window in here
    return profile_d
                       
   
def anno_vs_score_eval(antn, scores_d):
    collect_cm = {}
    collect_roc = {}
    
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # thresholds = [x+0.015 for x in thresholds]
    
    for thresh in thresholds:
        anom_instances = []
        norm_instances = []
        anom_res = []
        norm_res = []
        anno_sgm_d = {}

        # thresh = float((thresh/10)+0.5)
        for lbl, scores in scores_d.items(): #TODO note this is extremely dependent on keys in dict
            scores = [float(i) for i in scores]
            cat, num = lbl.split('/')
            # print("\n[PRED_EVAL] class:{} - video:{}".format(cat, num))
            # /home/tomsch/Honours/Honours_Project/output/all-fc6/anomaly-videos/Burglary_Test/076
            if not "Normal" in cat:
                # print("\t[PRED EVAL] anomalous video")
                vid_anno = antn[antn[:, 0] == cat]
                vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
                # print(vid_anno)
                if len(vid_anno) == 0:
                    pass
                    # print("\t\t************************WARNING: annotation not found for {}/{}".format(cat, num))
                else:
                    n_frames = int(vid_anno[2])
                    anno_cum = []
                    res = False
                    for pt in range(3, len(vid_anno), 2):
                        if vid_anno[pt] != '-1':
                            start_anom = int(vid_anno[pt])
                            end_anom = int(vid_anno[pt+1])
                            res, anno_sgm = verify_anom_fc6(scores, n_frames, ((start_anom, end_anom)), thresh, prev_res=res)   
                            
                            anno_cum += anno_sgm
                            anom_instances.append(1) #every anom is an instance even if 2 anoms come from same video
                            if res: 
                                anom_res.append(1) 
                            else:
                                anom_res.append(0)
                        else:
                            # print("-1 found, skip next -1")
                            pass
                    anno_sgm_d[lbl] = anno_cum
                    
            else:
                #TODO when we have normal videos
                # print("\t[PRED EVAL] normal video")
                norm_instances.append(1)
                # print(thresh)
                violating = list(filter(lambda x: (x > thresh), scores)) #maybe use filter instead
                if (len(violating) == 0):
                    norm_res.append(1) 
                else: 
                    norm_res.append(0)
                anno_sgm_d[lbl] = np.zeros(len(scores)).tolist()


            #pos is anomalous
            #neg is normal    
            
        tp = sum(anom_res)/sum(anom_instances)
        # tp = 1
        fn = 1 - tp
        tn = sum(norm_res)/sum(norm_instances) #TODO change back to normal instances
        # tn = thresh
        fp = 1 - tn
        cm = np.asarray([[tp, fn],[fp, tn]])
        collect_cm[thresh] = cm
        
        # instances = anom_instances + norm_instances
        # results = anom_res + norm_res
        
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        collect_roc[thresh] = (tpr, fpr)
    
    # CALCULATE RES AT BEST THRESHOLD (for display of S or F marker)
    best_thresh = 0
    best_cm_sum = 0
    for t, mat in collect_cm.items():
        cm_sum = mat[0][0] + mat[1][1]
        if cm_sum > best_cm_sum:
            best_cm_sum = cm_sum
            best_thresh = t 
        
    print("BEST THRESHOLD: {}\ngetting result markers at that thresh ...".format(best_thresh))
    res_d = {}
    for lbl, scores in scores_d.items(): #TODO note this is extremely dependent on paths of file system (path must contain "anom" or "norm")
        scores = [float(i) for i in scores]
        cat, num = lbl.split('/')
        # print("\n[PRED_EVAL] class:{} - video:{}".format(cat, num))
        # /home/tomsch/Honours/Honours_Project/output/all-fc6/anomaly-videos/Burglary_Test/076
        if not "Normal" in cat:
            # print("\t[PRED EVAL] anomalous video")
            vid_anno = antn[antn[:, 0] == cat]
            vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
            # print(vid_anno)
            if len(vid_anno) == 0:
                print("\t\tWARNING: annotation not found for {}/{}".format(cat, num))
                res_d[lbl] = None
            else:
                n_frames = int(vid_anno[2])
                res = False
                for pt in range(3, len(vid_anno), 2):
                    if vid_anno[pt] != '-1':
                        start_anom = int(vid_anno[pt])
                        end_anom = int(vid_anno[pt+1])
                        res, anno_sgm = verify_anom_fc6(scores, n_frames, ((start_anom, end_anom)), best_thresh, prev_res=res)   
                    else:
                        # print("-1 found, skip next -1")
                        pass
                
                res_d[lbl] = res
        else:
            #TODO when we have normal videos
            # print("\t[PRED EVAL] normal video")
            norm_instances.append(1)
            # print(float(thresh/10))
            violating = list(filter(lambda x: (x > best_thresh), scores)) #maybe use filter instead
            if (len(violating) == 0): 
                norm_res.append(1) 
            else:
                norm_res.append(0)
            res_d[lbl] = (len(violating)==0)

    
    plt.rcParams['text.color'] = "black"
    plt.rcParams['axes.labelcolor'] = "black"
    plt.rcParams['xtick.color'] = "black"
    plt.rcParams['ytick.color'] = "black"
    plt.rcParams['axes.facecolor']='ffffff'
    plt.rcParams['savefig.facecolor']='ffffff'
    plt.rcParams.update({'font.size': 14})

    
    # cnt = 0
    # r = 0
    # figcm, axcm = plt.subplots(3,3,constrained_layout=True)
    # # figcm.constrained_layout()
    # for t, cmat in collect_cm.items():
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=CLASSES)
    #     disp.plot(cmap='plasma', ax=axcm[r][cnt % 3], colorbar=False) 
    #     axcm[r][cnt%3].title.set_text("threshold = {}".format(t))
    #     cnt+=1
    #     if cnt % 3 == 0:
    #         r += 1
            
    # figbestcm, axbestcm = plt.subplots(1,1)
    # dispbest = ConfusionMatrixDisplay(confusion_matrix=collect_cm[0.4], display_labels=CLASSES)
    # dispbest.plot(cmap='Blues', ax=axbestcm, colorbar=True)
    
    
      
    # figroc, axroc = plt.subplots(1)
    # tprs = [x[0] for i, x in collect_roc.items()]
    # fprs = [x[1] for i, x in collect_roc.items()]
    # print(fprs)
    # print(tprs)
    # axroc.plot(fprs, tprs)
    # axroc.set_title("ROC Curve")  
    # axroc.set_xlim([0,1])
    # axroc.set_ylim([0,1])
    
    # axroc.set_xlabel("False Positive Rate")
    # axroc.set_ylabel("True Positive Rate")
    

    # fpr = {}
    # tpr = {}
    # roc_auc = {}
    # for i, s in collect_roc.items():
        # fpr[i], tpr[i], _ = roc_curve(s[0], s[1])
        # roc_auc[i] = auc(fpr[i], tpr[i])    
    
    # plt.show()
    
    # fig, ax = plt.subplots(1, 1)
    # disp = ConfusionMatrixDisplay(confusion_matrix=collect_cm[0.5], display_labels=CLASSES)
    # disp.plot(cmap='plasma', ax=ax, colorbar=False) 

    #TODO calc ROC and AUC and return as well
    return None, anno_sgm_d, res_d


def st_demo(combine_d, scores_d, sims_d, lkkm_d, anno_sgm_d, res_d, cm_disp, roc_disp, logo, lkkm_raw, lkkm_dcmp, raft_gunnar, consense, base_mod, pseudo):
    st.markdown("<h1 style='text-align: center; color: white; font-size: 3vw'>Anomaly Detection Framework</h1>", unsafe_allow_html=True)
    st.markdown("### A consensus framework for robust anomaly detection in CCTV surveillance.")
    st.markdown("<h1 style='text-align: center; color: white; font-size: 2vw'>Consensus</h1>", unsafe_allow_html=True)
    st.markdown("Score profiles are consensus between 3 models/heuristics, namely: *MIL base-model*, *CRAFT-flow*, *LKKM-flow*.")
    st.image(consense)
    st.markdown("The figure below displays pseudo-code of the alogorithm that is used to map 3 scores to 1 consensus score.")
    st.image(pseudo)
    
    st.markdown("<h1 style='text-align: center; color: white; font-size: 2vw'>Base Model</h1>", unsafe_allow_html=True)
    """
    The approach of the base model is adopted from *Real-world Anomaly Detection in Surveillance Videos by W. Sultani*. \n
    The base model formulates anomaly detection as a weakly-supervised regression problem that receives C3D features as predictors and produces a score in the range [0, 1] as the response. 
    """
    st.image(base_mod)
    
    
    st.markdown("<h1 style='text-align: center; color: white; font-size: 2vw'>Heuristic Experiments</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size: 1.5vw'>CRAFT: Truck/Train Collision</h1>", unsafe_allow_html=True)
    """
    The video clip below demonstrates CRAFT's exploitation of RAFT's inability to predict optical flow for anomalous sections of video. \n
    CRAFT quantifies anomaly by assigning anomaly scores proportional to reconstruction error. 
    """
    st_player("https://www.youtube.com/watch?v=gzedXyJQ7nQ")
    """
    The score profiles below were obtained during experimentation with the CRAFT concept. \n
    A column corresponds to a video instance. \n
    The top row of scores were extracted using RAFT for optical flow estimation.
    The bottom row of scores were extracted using the traditional Gunnar Farneback approach. 
    """
    st.image(raft_gunnar)
    
    st.markdown("<h1 style='text-align: center; color: white; font-size: 1.5vw'>LKKM: Highway U-Turn</h1>", unsafe_allow_html=True)
    """
    The video clip below is a demonstration of application of the Lukas-Kanade optical flow method to a highway scene that contains an accident. \n
    The intenion is to show that LKKM can be particularly useful in such a scenario, where a deviation from typical trajectories often corresponds to anomaly. 
    """
    st_player("https://www.youtube.com/watch?v=vVCYlmVoPeI")
    """
    The score profile below was extracted by applying LKKM to the above footage. \n
    That is, at each frame flow vectors contribute to a cumulative K-Means clustering of all flow vectors. \n
    Anomaly scores at each frame are proportional to the maximum distances between a number of flow vectors and their nearest clusters.  
    """
    st.image(lkkm_raw)
    st.markdown("<h1 style='text-align: center; color: white; font-size: 1.5vw'>LKKM: Repetitive Walk</h1>", unsafe_allow_html=True)
    """
    This video comprises repetitions of a short video clip of people walking on public walkways. \n
    LKKM is applied to the footage to learn repetitive patterns of motion. \n
    The intention is that LKKM produces incremental improvements in LKKM scores - this is an indication that with each trial of the same motion, LKKM increases the normal connotation attached to that motion. 
    """
    st_player("https://www.youtube.com/watch?v=yNtthVwohPA")
    """
    To address the fact that score profiles are likely to gradually decline as more footage is processed, 
    a time-series decomposition is performed on the scores and the score profile is replaced with the residual component of the decomposition.  \n
    For the repetitive walk, LKKM produces the score profile displayed at the top of the figure. The remaining score profiles are the components that result from decomposition. \n
    The bottom score profile is the residual component. Ideally, this profile captures chnages in LKKM scores that originate as a result of inherent noise in the time-series i.e., anomalous activity. 
    """
    st.image(lkkm_dcmp)
    
    st.markdown("<h1 style='text-align: center; color: white; font-size: 2vw'>Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size: 1.5vw'>ROC Curve</h1>", unsafe_allow_html=True)
    st.image(roc_disp, width=None)
    st.markdown("<h1 style='text-align: center; color: white; font-size: 1.5vw'>Optimal Confusion Matrix</h1>", unsafe_allow_html=True)
    st.image(cm_disp, width=None)
    
    st.markdown("<h1 style='text-align: center; color: white; font-size: 2vw'>Score Profiles</h1>", unsafe_allow_html=True)   
    """
    At each video, the display of relevant scores can be toggled and an option to view the video is presented.\n
    The categories of video include:
    """
    """
    * Abuse
    * Arrest
    * Arson
    * Assault
    * Burgalry
    * Explosion
    * Fighting
    * Road Accidents 
    * Robbery
    * Shooting
    * Shoplifting
    * Stealing
    * Vandalism
    * Normal
    """
    
    
    # st.write(scores_d.keys())
    for lbl in sorted(download_map.keys(), reverse=True):
        # st.write(vid)
        cat, num = lbl.split("/")       
        exp = st.expander(label=lbl, expanded=False)

        with exp:
#             dlkey = "{}{}".format(cat, num) 
            st.write("[Playback Video]({})".format(download_map[lbl]))
            # st_player("{}".format(download_map[lbl]))
                
            if lbl in combine_d.keys():
                combine_data = pd.DataFrame({
                    "final scores": combine_d[lbl],
                })
            else:
                combine_data = pd.DataFrame({
                    "final scores": np.zeros(1), 
                })
                print("[PRED EVAL] <WARNING> no matching label for this video in COMBINE dict.")
            
            if lbl in sims_d.keys():
                flow_data = pd.DataFrame({
                    "flow scores": sims_d[lbl],
                })
            else:
                flow_data = pd.DataFrame({
                    "flow scores": np.zeros(1), 
                })
                print("[PRED EVAL] <WARNING> no matching label for this video in FLOW dict.")
                
            if lbl in lkkm_d.keys():
                # print("is in lkkm_d.keys()******")
                lkkm_data = pd.DataFrame({
                    "lkkm scores": lkkm_d[lbl],
                })
            else:
                lkkm_data = pd.DataFrame({
                    "lkkm scores": np.zeros(1), 
                })
                print("[PRED EVAL] <WARNING> no matching label for this video in LKKM dict.")
                
            if lbl in scores_d.keys():
                mil_data = pd.DataFrame({
                    "MIL scores": scores_d[lbl],
                })
            else:
                mil_data = pd.DataFrame({
                    "MIL scores": np.zeros(1), 
                })
                print("[PRED EVAL] WARNING no matching label for this video in MIL scores dict.")    
                
            res = None
            anno_dat = np.zeros(len(scores_d[lbl])).tolist()
            if lbl in anno_sgm_d.keys():
                # print("is in anno_sgm.keys()*********")
                anno_sgms = anno_sgm_d[lbl]
                # anno_dat = np.squeeze(anno_dat)
                if cat != "Normal":
                    # print("anno_sgms")
                    # print(anno_sgms)
                    for a in anno_sgms:
                        anno_dat[int(a)] = 1 #1 indexing

                    # print(anno_dat)
                    set1 = False
                    for a in range(len(anno_dat)):
                        # print(a, end=" ")
                        if not set1:
                            if anno_dat[int(a)] == 1:
                                set1 = True
                        else:
                            if anno_dat[int(a)] == 1:
                                set1 = False
                            else:
                                anno_dat[int(a)] = 1
                    # print(anno_dat)
                    # print(len(anno_dat))
                else:
                    anno_dat[0] = 0 #to ensure that y-ax is scaled to 1 for normal footage
                    
                anno_data = pd.DataFrame({
                    "annotations": anno_dat,
                })
            else:
                anno_data = pd.DataFrame({
                    "annotations": anno_dat,
                })
                print("[PRED EVAL] <WARNING> no matching label for this video in ANNO dict.")
            
            all_data = anno_data 
            
            st.checkbox("Display MIL", value=False, key="{}{}MIL".format(cat, num)) 
            st.checkbox("Display CRAFT", value=False, key="{}{}FLOW".format(cat, num))
            st.checkbox("Display LKKM", value=False, key="{}{}LKKM".format(cat, num))
            st.checkbox("Display FINAL", value=True, key="{}{}FINAL".format(cat, num))

            if eval("st.session_state.{}{}MIL".format(cat, num)):
                # all_data = pd.concat([all_data, mil_data], axis=1, ignore_index=True)              
                all_data["base-score"] = mil_data
            if eval("st.session_state.{}{}FLOW".format(cat, num)):
                all_data["craft-score"] = flow_data
            if eval("st.session_state.{}{}LKKM".format(cat, num)):
                all_data["lkkm-score"] = lkkm_data
            if eval("st.session_state.{}{}FINAL".format(cat, num)):
                all_data["consensus"] = combine_data
                            
            # fig = plt.figure(figsize=(10, 4))
            # years_of_experience =[1,2,3]
            # salary=[ [6,8,10], [4,5,9], [3,5,7] ]
            # plt.stackplot(all_data, labels=['Company A','Company B','Company C'])
            # plt.title('Chart title')
            # plt.xlabel('X axis title')
            # plt.ylabel('Y axis title') 
            # plt.legend(loc='upper left')

            # sns.lineplot(x = "distance", y = "mass", data = data_frame)
            # st.pyplot(fig)
            # custom_chart = alt.Chart(all_data).mark_area().encode(
            #     x='temporal axis',
            #     y='anomaly score',
            #     color=alt.Color('animal',
            #     scale=alt.Scale(
            #     domain=['antelope', 'velociraptor'],
            #     range=['blue', 'red'])
            #     )).properties(
            #         width=900,
            #         height=500)

            # st.altair_chart(custom_chart)
            st.area_chart(all_data, use_container_width=True)
            

def consensus(base, sims, lkkm, wdw):
    profile_d = {}
    # print(len(hrts.keys()))
    for lbl, b in base.items():
        s = sims[lbl]
        if lbl == "RoadAccidents/021":
            l = sims[lbl]
        else:
            l = lkkm[lbl]
        c = []
        assert len(s) == 32
        assert len(b) == 32
        assert len(l) == 32
        
        for i in range(32):
            # if i > wdw and (len(b)-i-1) > wdw:
            #     fg = sum(h[i-wdw:i+wdw])
            #     cg = sum(b[i-wdw:i+wdw])
            #     fg /= wdw
            #     cg /= wdw

            # elif i < wdw:
            #     fg = sum(h[:i])
            #     cg = sum(b[:i])
            #     fg /= max(1,i)
            #     cg /= max(1,i)

            # elif (len(b)-i-1) < wdw:
            #     fg = sum(h[i:])
            #     cg = sum(b[i:])
            #     fg /= max(1,i)
            #     cg /= max(1,i)
            if b[i] == 0:
                c.append(b[i]) 
            else:
                c.append(max([b[i], s[i], l[i]]))
                # c.append(0.5*(1+h[i])*b[i])
        profile_d[lbl] = c
        assert len(c) == 32
    return profile_d    

def main(pred_path, sim_path, lkkm_path):
    antn = read_anno_file(ANNO_FILE)
    with open(pred_path, "r") as f:
        scores = json.load(f)
        scores_d = json.loads(scores)
    with open(sim_path, "r") as f:
        sims = json.load(f)
        sims_d = json.loads(sims)
    with open(lkkm_path, "r") as f:
        lkkm = json.load(f)
        lkkm_d = json.loads(lkkm)
    # with open(delta_path, "r") as f:
        # deltas = json.load(f)
        # deltas_d = json.loads(deltas)

    
    #CRAFT process
    sims_d = convert_to_32(sims_d)
    lkkm_d = decompose(lkkm_d)
    lkkm_d = convert_to_32(lkkm_d)
    
    #indiv process
    sims_d = individual_scale(sims_d)
    sims_d = filter_to_zero(sims_d, 1.5)
    sims_d = nan_to_zero(sims_d)
    sims_d = to_range(sims_d, 1)
    sims_d = apply_broadc(sims_d, 16)
    
    lkkm_d = individual_scale(lkkm_d)
    lkkm_d = filter_to_zero(lkkm_d, 1.5)
    lkkm_d = nan_to_zero(lkkm_d)
    lkkm_d = to_range(lkkm_d, 1)
    lkkm_d = apply_broadc(lkkm_d, 16)
    # hrts_d = combine_heuristics(sims_d, lkkm_d)
    
    scores_d = nan_to_zero(scores_d)
    scores_d = filter_to_zero(scores_d, 0.3)
    scores_d = apply_broadc(scores_d, 16)
    # scores_d = to_range(scores_d, 1)
    
    profile_d = consensus(scores_d, sims_d, lkkm_d, 1)

    # hrts_d = smoothing(hrts_d, 8)
    
    #SEQUENCE
    # anno_seq, score_seq = compute_score_seq(scores_d, antn)
    # anno_seqp, profile_seq = compute_score_seq(profile_d, antn)
    # anno_seql, lkkm_seq = compute_score_seq(lkkm_d, antn)
    # anno_seqc, sims_seq = compute_score_seq(sims_d, antn)
    
    #ROC/AUC
    # figroc, ax = plt.subplots(1,1)
    # 
    # ax.plot([0,1], [0,1], label='Binary SVM: AUC=0.5', linestyle="dashed", color="black", linewidth=3)
    # 
    # fprb, tprb, threshb = roc_curve(anno_seq, score_seq)
    # aucb = roc_auc_score(anno_seq, score_seq)
    # 
    # ax.plot(fprb,tprb,label="Base: AUC="+str(round(aucb,4)), color="blue",linewidth=3)
    # 
    # fprc, tprc, threshc = roc_curve(anno_seqc, sims_seq)
    # aucc = roc_auc_score(anno_seqc, sims_seq)
    # ax.plot(fprc,tprc,label="CRAFT: AUC="+str(round(aucc,4)), color="purple", linewidth=3)
# 
    # fprl, tprl, threshl = roc_curve(anno_seql, lkkm_seq)
    # aucl = roc_auc_score(anno_seql, lkkm_seq)
    # ax.plot(fprl,tprl,label="LKKM: AUC="+str(round(aucl,4)), color="orange", linewidth=3)
    # 
    # fprp, tprp, threshp = roc_curve(anno_seqp, profile_seq)
    # aucp = roc_auc_score(anno_seqp, profile_seq)
    # ax.plot(fprp,tprp,label="Consensus: AUC="+str(round(aucp,4)), color="red",linewidth=3)
    # 
    # plt.legend(loc=0, fontsize=13)
    # plt.xlabel('False Positive Rate', fontsize=14)
    # plt.ylabel('True Positive Rate', fontsize=14)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)    
    # plt.show()
    
    # ConfusionMatrixDisplay.from_predictions(anno_seq, score_seq, normalize='true')
    # plt.show()
    
    # thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # thresholds = [x+0.05 for x in thresholds]

    # for thresh in thresholds:
    #     score_seq_class = []
    #     for i in range(len(profile_seq)):
    #         if profile_seq[i] > thresh:
    #             score_seq_class.append(1)
    #         else:
    #             score_seq_class.append(0)
    #     print(f"THRESH: {thresh}")
    #     print(anno_seq)
    #     print(score_seq)
    #     print(score_seq_class)     
    #     ConfusionMatrixDisplay.from_predictions(anno_seq, score_seq_class, normalize='true')
    #     plt.show()
    
    cm_disp, anno_sgm_d, res_d = anno_vs_score_eval(antn, scores_d) #TODO back to profile
    cm_img =  Image.open("demo/img/CMsb.png")
    roc_img =  Image.open("demo/img/ROCsb.png")
    logo = Image.open("demo/img/logo.png")
    lkkm_raw = Image.open("demo/img/lkkm_raw.png")
    lkkm_dcmp = Image.open("demo/img/lkkm_dcmp.png")
    raft_gunnar = Image.open("demo/img/raft_gunnar.png")
    base_mod = Image.open("demo/img/basemodel.png")
    consense = Image.open("demo/img/consense.png")
    pseudo = Image.open("demo/img/pseudocomb.png")
    
    st_demo(combine_d=profile_d, #TODO should be profile_d
        scores_d=scores_d,
        sims_d=sims_d,
        lkkm_d=lkkm_d,
        anno_sgm_d=anno_sgm_d,
        res_d=res_d,
        cm_disp=cm_img,
        roc_disp=roc_img,
        logo = logo,
        lkkm_raw = lkkm_raw,
        lkkm_dcmp=lkkm_dcmp,
        raft_gunnar=raft_gunnar,
        consense=consense,
        base_mod=base_mod,
        pseudo=pseudo
        )
        
if __name__ == '__main__':
    pred_path = "ofc-data/kraken_base_scores.json"
    sim_path = "ofc-data/kraken_craft_sim.json"
    lkkm_path = "ofc-data/kraken_lkkm_ncd.json"
    # 
    # test_data = [0.0013, 0.0032, 0.0, 0.2973, 0.0101, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0049, 0.3753, 0.0015, 0.005, 0.4771, 0.0, 0.0, 0.5081, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0003, 0.009, 0.0014, 0.0, 0.8922, 0.0053, 0.0, 0.0001, 0.0002, 0.0, 0.0003, 0.0, 0.147, 0.0291, 0.0032, 0.0191, 0.0, 0.0, 0.0775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0101, 0.439, 0.0003, 0.0034, 0.9953, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0021, 0.0006, 0.0071, 0.097, 0.0, 0.1574, 0.0084, 0.0, 0.0001, 0.0, 0.0, 0.0006, 0.0, 0.0002, 0.0006, 0.0002, 0.6699, 0.0, 0.0, 0.9287, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.001, 0.0179, 0.0002, 0.9951, 0.0, 0.0, 0.0008, 0.0001, 0.0, 0.0, 0.0, 0.0007, 0.0004, 0.0032, 0.0408, 0.0, 0.0056, 0.017, 0.0, 0.0, 0.0, 0.0, 0.0012, 0.0032]
   
    main(pred_path, sim_path, lkkm_path)
    
