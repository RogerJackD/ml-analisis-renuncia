// Global variables for model and scaler
let model;

// Feature names in the exact order the model expects them
const featureOrder = [
    'Age',                  // 1
    'DailyRate',            // 2
    'DistanceFromHome',     // 3
    'Education',            // 4
    'EnvironmentSatisfaction', // 5
    'HourlyRate',           // 6
    'JobInvolvement',       // 7
    'JobLevel',             // 8
    'JobSatisfaction',      // 9
    'MonthlyIncome',        // 10 - ¡IMPORTANTE! Esta estaba faltando
    'MonthlyRate',          // 11
    'NumCompaniesWorked',   // 12
    'PercentSalaryHike',    // 13
    'PerformanceRating',    // 14
    'RelationshipSatisfaction', // 15
    'StockOptionLevel',     // 16
    'TotalWorkingYears',    // 17
    'TrainingTimesLastYear', // 18
    'WorkLifeBalance',      // 19
    'YearsAtCompany',       // 20
    'YearsInCurrentRole',   // 21
    'YearsSinceLastPromotion', // 22
    'YearsWithCurrManager', // 23
    'BusinessTravel_Travel_Frequently', // 24
    'BusinessTravel_Travel_Rarely', // 25
    'Department_Research & Development', // 26
    'Department_Sales',     // 27
    'EducationField_Life Sciences', // 28
    'EducationField_Marketing', // 29
    'EducationField_Medical', // 30
    'EducationField_Other', // 31
    'EducationField_Technical Degree', // 32
    'Gender_Male',         // 33
    'JobRole_Healthcare Representative', // 34 - ¡Faltaba esta!
    'JobRole_Human Resources', // 35
    'JobRole_Laboratory Technician', // 36
    'JobRole_Manager',      // 37
    'JobRole_Manufacturing Director', // 38
    'JobRole_Research Director', // 39
    'JobRole_Research Scientist', // 40
    'JobRole_Sales Executive', // 41
    'JobRole_Sales Representative', // 42
    'MaritalStatus_Married', // 43
    'MaritalStatus_Single', // 44
    'OverTime_Yes',        // 45
    'EmployeeCount',       // 46 - Generalmente es 1
    'EmployeeNumber'       // 47 - Podemos usar 0 como default
];

// Load model when page loads
async function loadModel() {
    try {
        // Load the TensorFlow.js model
        model = await tf.loadLayersModel('model_attrition_web/model.json');
        console.log("Model loaded successfully");
        
        // Verificar que el modelo tiene la forma esperada
        console.log("Model input shape:", model.inputs[0].shape);
    } catch (error) {
        console.error("Error loading model:", error);
        alert("Error loading the prediction model. Please check console for details.");
    }
}

// Call the load function when the page loads
document.addEventListener('DOMContentLoaded', loadModel);

// Function to preprocess input data
function preprocessInput(data) {
    // 1. Inicializar todas las características a 0
    let features = {};
    featureOrder.forEach(feat => features[feat] = 0);
    
    // 2. Características numéricas del formulario
    features['Age'] = parseFloat(data.age) || 30;
    features['MonthlyIncome'] = parseFloat(data.monthlyIncome) || 5000;
    features['TotalWorkingYears'] = parseFloat(data.totalWorkingYears) || 5;
    features['YearsAtCompany'] = parseFloat(data.yearsAtCompany) || 3;
    
    // 3. Valores por defecto para otras numéricas
    features['DailyRate'] = 800;
    features['DistanceFromHome'] = 10;
    features['Education'] = 3;
    features['EnvironmentSatisfaction'] = 3;
    features['HourlyRate'] = 65;
    features['JobInvolvement'] = 3;
    features['JobLevel'] = 2;
    features['JobSatisfaction'] = 3;
    features['MonthlyRate'] = 20000;
    features['NumCompaniesWorked'] = 2;
    features['PercentSalaryHike'] = 15;
    features['PerformanceRating'] = 3;
    features['RelationshipSatisfaction'] = 3;
    features['StockOptionLevel'] = 0;
    features['TrainingTimesLastYear'] = 3;
    features['WorkLifeBalance'] = 3;
    features['YearsInCurrentRole'] = Math.floor(features['YearsAtCompany'] / 2);
    features['YearsSinceLastPromotion'] = Math.floor(features['YearsAtCompany'] / 3);
    features['YearsWithCurrManager'] = Math.floor(features['YearsAtCompany'] / 2);
    features['EmployeeCount'] = 1;  // Generalmente es 1
    features['EmployeeNumber'] = 0; // No es relevante para la predicción
    
    // 4. Variables categóricas del formulario
    features['OverTime_Yes'] = data.overTime === 'Yes' ? 1 : 0;
    
    // Business Travel
    features['BusinessTravel_Travel_Frequently'] = data.businessTravel === 'Travel_Frequently' ? 1 : 0;
    features['BusinessTravel_Travel_Rarely'] = data.businessTravel === 'Travel_Rarely' ? 1 : 0;
    
    // Department
    features['Department_Research & Development'] = 1;
    features['Department_Sales'] = 0;
    
    // Education Field
    features['EducationField_Life Sciences'] = 1;
    features['EducationField_Marketing'] = 0;
    features['EducationField_Medical'] = 0;
    features['EducationField_Other'] = 0;
    features['EducationField_Technical Degree'] = 0;
    
    // Gender
    features['Gender_Male'] = 1;
    
    // Job Role
    const jobRoleMap = {
        'Sales Executive': 'JobRole_Sales Executive',
        'Research Scientist': 'JobRole_Research Scientist',
        'Laboratory Technician': 'JobRole_Laboratory Technician',
        'Manufacturing Director': 'JobRole_Manufacturing Director',
        'Healthcare Representative': 'JobRole_Healthcare Representative',
        'Manager': 'JobRole_Manager',
        'Sales Representative': 'JobRole_Sales Representative',
        'Research Director': 'JobRole_Research Director',
        'Human Resources': 'JobRole_Human Resources'
    };
    
    // Reset all job roles
    Object.values(jobRoleMap).forEach(role => {
        features[role] = 0;
    });
    
    // Set selected job role
    if (jobRoleMap[data.jobRole]) {
        features[jobRoleMap[data.jobRole]] = 1;
    }
    
    // Marital Status
    features['MaritalStatus_Married'] = 0;
    features['MaritalStatus_Single'] = 1;
    
    // 5. Convertir a array y verificar
    let featureArray = featureOrder.map(feat => {
        if (features[feat] === undefined) {
            console.error(`Missing feature in processing: ${feat}`);
            return 0;
        }
        return features[feat];
    });
    
    // Verificación crítica
    if (featureArray.length !== 47) {
        console.error(`Error: Generated ${featureArray.length} features, expected 47`);
        console.log("Generated features:", featureArray);
        throw new Error(`Feature count mismatch. Generated ${featureArray.length}, expected 47`);
    }
    
    return featureArray;
}

// Function to make prediction
async function predictAttrition() {
    if (!model) {
        alert("Model is still loading. Please wait a moment and try again.");
        return;
    }
    
    try {
        // Verify all required elements exist
        const requiredElements = ['age', 'monthlyIncome', 'jobRole', 'totalWorkingYears', 
                                'yearsAtCompany', 'overTime', 'businessTravel'];
        const missingElements = requiredElements.filter(id => !document.getElementById(id));
        
        if (missingElements.length > 0) {
            throw new Error(`Missing form elements: ${missingElements.join(', ')}`);
        }
        
        // Get input values
        const inputData = {
            age: document.getElementById('age').value,
            monthlyIncome: document.getElementById('monthlyIncome').value,
            jobRole: document.getElementById('jobRole').value,
            totalWorkingYears: document.getElementById('totalWorkingYears').value,
            yearsAtCompany: document.getElementById('yearsAtCompany').value,
            overTime: document.getElementById('overTime').value,
            businessTravel: document.getElementById('businessTravel').value
        };
        
        // Preprocess the input
        const processedInput = preprocessInput(inputData);
        
        // Verify feature count
        if (processedInput.length !== 47) {
            throw new Error(`Feature count mismatch. Expected 47, got ${processedInput.length}`);
        }
        
        // Convert to tensor and predict
        const inputTensor = tf.tensor2d([processedInput], [1, 47]);
        const prediction = model.predict(inputTensor);
        const probability = (await prediction.data())[0];
        
        // Display result
        displayResult(probability);
        
        // Clean up
        inputTensor.dispose();
        prediction.dispose();
        
    } catch (error) {
        console.error("Prediction error:", error);
        alert(`Prediction failed: ${error.message}. Check console for details.`);
    }
}

// Function to display prediction result
function displayResult(probability) {
    const resultDiv = document.getElementById('result');
    const percentage = (probability * 100).toFixed(2);
    
    resultDiv.style.display = 'block';
    resultDiv.textContent = `Probability of attrition: ${percentage}%`;
    
    if (probability > 0.5) {
        resultDiv.className = 'result high-risk';
        resultDiv.textContent += ' - HIGH risk of attrition';
    } else {
        resultDiv.className = 'result low-risk';
        resultDiv.textContent += ' - LOW risk of attrition';
    }
}