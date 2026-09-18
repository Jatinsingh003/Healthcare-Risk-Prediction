create database healthcare


-- drop table diagnosis
-- drop table outcomes

create table diagnosis(
      diagnosis_id int primary key,
      diagnosis_name varchar(225)
);

create table outcomes(
      outcome_id int primary key,
      outcome_name varchar(225)
);

create table patients(
       patient_id int primary key,
       name varchar(225),
       age int,
       gender char,
       diagnosis_id int,
       admission_date date,
       discharge_date date,
       outcome_id int,
       treatment_cost numeric(10,2),
       foreign key (diagnosis_id) references diagnosis(diagnosis_id),
       foreign key (outcome_id) references outcomes(outcome_id)
);

create table labs(
       lab_id int primary key,
       patient_id int,
       test_name varchar(225),
       result numeric(10,2),
       normal_range varchar(225),
       foreign key (patient_id) references patients(patient_id)
);

select * from diagnosis;
select * from outcomes;
select * from patients;
select * from labs;

ALTER TABLE diagnosis
RENAME TO diagnosis;

ALTER TABLE diagnosis
RENAME COLUMN diagnoses_id TO diagnosis_id;

ALTER TABLE diagnosis
RENAME COLUMN diagnoses_name TO diagnosis_name;

ALTER TABLE patients
RENAME COLUMN diagnoses_id TO diagnosis_id;

ALTER TABLE patients
RENAME COLUMN treatement_cost TO treatment_cost;


-- Retrieving detailed patients lab history.
SELECT p.patient_id, p.name, d.diagnosis_name, o.outcome_name, 
       l.test_name, l.result, l.normal_range
FROM patients p
JOIN diagnosis d 
     ON d.diagnosis_id = p.diagnosis_id
JOIN outcomes o
     ON o.outcome_id = p.outcome_id
JOIN labs l
     ON l.patient_id = p.patient_id
ORDER BY p.patient_id, l.test_name;

--Average lab results by diagnosis
SELECT d.diagnosis_name, l.test_name, avg(l.result)::NUMERIC(10,2) AS Average_results, 
       count(p.patient_id):: int AS No_of_patients
FROM diagnosis d
JOIN  patients p
     ON d.diagnosis_id = p.diagnosis_id
JOIN labs l
     ON p.patient_id = l.patient_id
GROUP BY d.diagnosis_name, l.test_name
ORDER BY No_of_patients DESC;

--Abnormal lab results.
--(taking idea about range of normal results from ai as i dont have medical knowledge)
WITH lab_status AS (
    SELECT
        p.patient_id, p.name AS patient_name,
        CASE
            WHEN l.test_name = 'Systolic BP'
                 AND l.result >= 120 THEN 'Abnormal'
            WHEN l.test_name = 'Diastolic BP'
                 AND l.result >= 80 THEN 'Abnormal'
            WHEN l.test_name = 'Cholesterol'
                 AND (l.result < 125 OR l.result > 200) THEN 'Abnormal'
            WHEN l.test_name = 'HbA1c'
                 AND l.result >= 5.7 THEN 'Abnormal'
            WHEN l.test_name = 'Fasting Glucose'
                 AND (l.result < 70 OR l.result > 99) THEN 'Abnormal'
            WHEN l.test_name = 'BMI'
                 AND (l.result < 18.5 OR l.result > 24.9) THEN 'Abnormal'
            WHEN l.test_name = 'WBC Count'
                 AND (l.result < 4 OR l.result > 11) THEN 'Abnormal'
            WHEN l.test_name = 'CRP'
                 AND l.result > 5 THEN 'Abnormal'
            WHEN l.test_name = 'Oxygen Saturation'
                 AND l.result < 95 THEN 'Abnormal'
            WHEN l.test_name = 'Peak Flow'
                 AND l.result < 400 THEN 'Abnormal'
            WHEN l.test_name = 'Respiratory Rate'
                 AND (l.result < 12 OR l.result > 20) THEN 'Abnormal'
            WHEN l.test_name = 'Troponin I'
                 AND l.result > 0.04 THEN 'Abnormal'
            WHEN l.test_name = 'LDL Cholesterol'
                 AND l.result >= 100 THEN 'Abnormal'
            WHEN l.test_name = 'HDL Cholesterol'
                 AND (l.result < 40 OR l.result > 60) THEN 'Abnormal'
            WHEN l.test_name = 'Creatinine'
                 AND (l.result < 0.60 OR l.result > 1.30) THEN 'Abnormal'
            WHEN l.test_name = 'Migraine Severity'
                 AND l.result > 5 THEN 'Abnormal'
            WHEN l.test_name = 'Blood Pressure'
                 AND l.result >= 120 THEN 'Abnormal'
            WHEN l.test_name = 'Amylase'
                 AND (l.result < 30 OR l.result > 110) THEN 'Abnormal'
            WHEN l.test_name = 'Hemoglobin'
                 AND (l.result < 12 OR l.result > 16) THEN 'Abnormal'
            WHEN l.test_name = 'Ferritin'
                 AND (l.result < 15 OR l.result > 150) THEN 'Abnormal'
            WHEN l.test_name = 'Platelet Count'
                 AND (l.result < 150 OR l.result > 450) THEN 'Abnormal'
            WHEN l.test_name = 'Temperature'
                 AND (l.result < 36.1 OR l.result > 37.2) THEN 'Abnormal'
            ELSE 'Normal'
        END AS result_status
    FROM patients AS p
    JOIN labs AS l
        ON p.patient_id = l.patient_id
)
SELECT
    patient_id,
    patient_name,
    CASE
        WHEN COUNT(*) FILTER (WHERE result_status = 'Abnormal') > 0
            THEN 'Abnormal'
        ELSE 'Normal'
    END AS status,
    COUNT(*) FILTER (WHERE result_status = 'Normal') AS normal_count,
    COUNT(*) FILTER (WHERE result_status = 'Abnormal') AS abnormal_count
FROM lab_status
GROUP BY patient_id, patient_name
ORDER BY patient_id;


--Highest Treatment cost(diagnosis)
SELECT d.diagnosis_name,
       sum(p.treatment_cost) AS Total_cost
FROM patients p
JOIN  diagnosis d
     ON d.diagnosis_id = p.diagnosis_id
JOIN labs l
     ON p.patient_id = l.patient_id
GROUP BY d.diagnosis_name
ORDER BY Total_cost DESC;

--Patients at risk by Outcome_name
SELECT p.name AS Patient_name, o.outcome_name,
       CASE 
	       WHEN o.outcome_name = 'Referred' 
		        THEN 'At Risk-Require Attention'
		   WHEN o.outcome_name = 'Deceased'
                THEN 'Critical'
				ELSE 'No Risk'
		   END AS "Risk Status"

FROM patients p
JOIN outcomes o
     ON p.outcome_id = o.outcome_id
GROUP BY p.name, o.outcome_name;

--Patients at risk by age
SELECT p.name AS Patient_name, o.outcome_name, p.age,
       CASE 
	       WHEN o.outcome_name = 'Referred' AND age > 65 OR 
		        o.outcome_name = 'Deceased' AND age > 65 OR
				o.outcome_name = 'Referred' OR
                o.outcome_name = 'Deceased'
		        THEN 'At Risk-Require Attention'
				ELSE 'No Risk'
				END AS "Risk Status"
FROM patients p
JOIN outcomes o
     ON p.outcome_id = o.outcome_id
GROUP BY p.name, o.outcome_name, p.age
ORDER BY p.age DESC;

--Distribution of outcomes by diagnosis with their count
SELECT d.diagnosis_name, o.outcome_name, count(o.outcome_name) AS Outcome_count
FROM diagnosis d
JOIN patients p
     ON d.diagnosis_id = p.diagnosis_id
JOIN outcomes o
     ON o.outcome_id = p.outcome_id
JOIN labs l
     ON l.patient_id = p.patient_id
GROUP BY d.diagnosis_name, o.outcome_name
ORDER BY d.diagnosis_name;







	