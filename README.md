# CSCE421_Final_Project
1. Create new dataframe with desired structure called data (there will potentially be 2016 rows)
    - column_names = [patientunitstayid, hospitaldischargestatus, cellattributevalue, ethnicity, gender, unitvisitnumber, admissionheight, admissionweight, age, glucose_avg, pH_avg, o2_saturation_avg, heart_rate_avg, ni_bp_dia_avg, ni_bp_sys_avg, ni_bp_mean_avg, respiratory_rate_avg, invasive_bp_dia_avg, invasive_bp_sys_avg, invasive_bp_mean_avg, gcs_total_avg]

2. Read in data from csv into a parsing dataframe (df)
#parsing dataframe has column names of [admissionheight, admissionweight, age, cellattributevalue, celllabel, ethnicity, gender, labmeasurenamesystem, labname, labresult, nursingchartcelltypevalname, nursingchartvalue, offset, patientunitstayid, unitvisitnumber]
- ethnicity = (caucasian, hispanic, asian, African American, Native American, Other/Unknown/empty) 
            = (0, 1, 2, 3, 4, 5)


#A there are only 2016 unique patients, so we are only reading the first 2016 lines to get patientunitstayid, ethnicity, gender, unitvisitnumber, admissionheight, admissionweight, admissionage. 
3. For row in range[0, 2015] 
    - initialize new_row with null values
    - add data (described at #A) into df[row]
    - for patient_data_rows in all rows with instance of patientunitstayid:
        a. cellattributevalue, cellabel(ignore)
            - take mode over all cellattributevalue (normal, <2s, >2s, feet, hands) -> (0, 1, 2, 3, 4) which one appears the most
            - put mode in (figure out syntax later) 
        b. labname, labresult
            - for each labname, take average of that labname from labresult data (ph, glucose)
            - put averages in (figure out syntax later)
        c. nursingchartcelltypevalname, nursingchartvalue
            - for each nursingchartcelltypevalname, take average of corresponding nursingchartvalue (O2 Saturation, Heart Rate, Respiratory Rate, Non-invasive BP Systolic, Non-invasive BP Diastolic, Non-invasive BP Mean, Invasive BP Diastolic, Invasive BP Systolic, Invasive BP Mean, GCS Total) 
            - put averages in (figure out syntax later)

4. Output data into a new csv
