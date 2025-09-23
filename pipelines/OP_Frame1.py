def bmd_diagnosis(age: int, gender: str, score_type: str, score: float, ) -> str:
    """
    Diagnose bone density status based on age, gender, and score (T-score or Z-score).

    :param age: Age in years (1-150)
    :param gender: 'Male', 'Female'
    :param score: BMD score (T-score or Z-score)
    :param score_type: 'T', 'Z',
    :return: Diagnosis string: 'Normal', 'Early Stage', or 'Disease'
    """
    # Normalize inputs
    gender = gender.lower().strip()
    score_type = score_type.lower().strip()
    is_female = gender in ['Female']
    is_male = gender in ['Male']

    # Auto-detect score type if 'auto'
   # if score_type == 'auto':
    #    score_type = 't' if age >= 50 else 'z'

    # T-score decision
    if score_type == 't-score':
        if score >= -1.0:
            return "Normal"
        elif -2.5 < score < -1.0:
            return "Osteopenia"
        else:  # score <= -2.5
            return "Osteoporosis"

    # Z-score decision
    elif score_type == 'z-score':
        if score >= -2.0:
            return "Normal"
        elif -3.0 < score < -2.0:
            return "Osteopenia"
        else:  # score <= -3.0
            return "Osteoporosis"
    else:
        return "Unknown"