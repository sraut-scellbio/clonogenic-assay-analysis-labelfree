area_coeffs = {
    "u87":{
        'mu': 62.61,
        'std': 23.81,
        'min': 10,
        'max': 128.96,
    },

    "u251":{
        'mu': 81.64,
        'std': 22.79,
        'min': 10,
        'max': 250,
    }
}

aspect_ratio_coeffs = {
    "u87": {
        'mu': 0.9713,
        'std': 0.16312,
        'min': 0.57,
        'max': 1.41,
        'qr1': 0.86, # adjusted from 1.07
        'qr3': 1.17
    },

    "u251": {
        'mu': 1.009,
        'std': 0.246,
        'min': 0.2,
        'max': 2.1,
        'qr1': 0.84, # adjusted from 1.07
        'qr3': 1.15
    }
}

distr_coeffs = {
    "u87": {
        'area_coeff': 1.75,
        'ar_coeff': 1.75
    },
    "u251": {
        'area_coeff': 1.55,
        'ar_coeff': 1.55
    }
}