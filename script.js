// Global variables for model and scaler
let model;
let scaler;

// Feature names in the exact order the model expects them
const featureOrder = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 
    'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 
    'JobLevel', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 
    'BusinessTravel_Travel_Rarely', 'Department_Research & Development', 
    'Department_Sales', 'EducationField_Life Sciences', 'EducationField_Marketing', 
    'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree', 
    'Gender_Male', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 
    'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative', 
    'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes'
];

// Load model and scaler when page loads
async function loadModelAndScaler() {
    try {
        // Load the TensorFlow.js model
        model = await tf.loadLayersModel('model_attrition_web/model.json');
        
        // Load the scaler (in a real app, you would need to implement the scaler logic)
        // For this example, we'll simulate it with a placeholder
        console.log("Model and scaler loaded successfully");
    } catch (error) {
        console.error("Error loading model or scaler:", error);
    }
}

// Call the load function when the page loads
document.addEventListener('DOMContentLoaded', loadModelAndScaler);

// Function to preprocess input data
// Function to preprocess input data
function preprocessInput(data) {
    // Create a dictionary with all features initialized to 0
    let features = {};
    featureOrder.forEach(feat => features[feat] = 0);
    
    // Set numerical features
    features['Age'] = parseFloat(data.age);
    features['MonthlyIncome'] = parseFloat(data.monthlyIncome);
    features['TotalWorkingYears'] = parseFloat(data.totalWorkingYears);
    features['YearsAtCompany'] = parseFloat(data.yearsAtCompany);
    
    // Set default values for other numerical features not in the form
    features['DailyRate'] = 800;  // Valor promedio aproximado
    features['DistanceFromHome'] = 10;
    features['Education'] = 3;
    features['EnvironmentSatisfaction'] = 3;
    features['JobInvolvement'] = 3;
    features['JobLevel'] = 2;
    features['JobSatisfaction'] = 3;
    features['NumCompaniesWorked'] = 2;
    features['PercentSalaryHike'] = 15;
    features['PerformanceRating'] = 3;
    features['RelationshipSatisfaction'] = 3;
    features['StockOptionLevel'] = 0;
    features['TrainingTimesLastYear'] = 2;
    features['WorkLifeBalance'] = 3;
    features['YearsInCurrentRole'] = Math.floor(parseFloat(data.yearsAtCompany) / 2);
    features['YearsSinceLastPromotion'] = Math.floor(parseFloat(data.yearsAtCompany) / 3);
    features['YearsWithCurrManager'] = Math.floor(parseFloat(data.yearsAtCompany) / 2);
    
    // Handle categorical variables from the form
    features['OverTime_Yes'] = data.overTime === 'Yes' ? 1 : 0;
    
    // Business Travel
    features['BusinessTravel_Travel_Frequently'] = data.businessTravel === 'Travel_Frequently' ? 1 : 0;
    features['BusinessTravel_Travel_Rarely'] = data.businessTravel === 'Travel_Rarely' ? 1 : 0;
    
    // Department (asumimos Research & Development como valor por defecto)
    features['Department_Research & Development'] = 1;
    features['Department_Sales'] = 0;
    
    // Education Field (asumimos Life Sciences como valor por defecto)
    features['EducationField_Life Sciences'] = 1;
    features['EducationField_Marketing'] = 0;
    features['EducationField_Medical'] = 0;
    features['EducationField_Other'] = 0;
    features['EducationField_Technical Degree'] = 0;
    
    // Gender (asumimos Male como valor por defecto)
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
    
    // Reset all job roles to 0 first
    featureOrder.filter(f => f.startsWith('JobRole_')).forEach(f => features[f] = 0);
    
    // Set the selected job role
    if (jobRoleMap[data.jobRole]) {
        features[jobRoleMap[data.jobRole]] = 1;
    }
    
    // Marital Status (asumimos Single como valor por defecto)
    features['MaritalStatus_Married'] = 0;
    features['MaritalStatus_Single'] = 1;
    
    // Convert the features object to an array in the correct order
    let featureArray = featureOrder.map(feat => features[feat]);
    
    return featureArray;
}

// Function to make prediction
async function predictAttrition() {
    if (!model) {
        alert("Model is still loading. Please wait a moment and try again.");
        return;
    }
    
    // Verificar que todos los elementos existen antes de continuar
    const requiredElements = [
        'age', 'monthlyIncome', 'jobRole', 'totalWorkingYears', 
        'yearsAtCompany', 'overTime', 'businessTravel'
    ];
    
    const missingElements = requiredElements.filter(id => !document.getElementById(id));
    
    if (missingElements.length > 0) {
        console.error("Missing elements:", missingElements);
        alert(`Error: Missing form elements (${missingElements.join(', ')}). Check your HTML.`);
        return;
    }
    
    // Get input values (solo si todos los elementos existen)
    const inputData = {
        age: document.getElementById('age').value,
        monthlyIncome: document.getElementById('monthlyIncome').value,
        jobRole: document.getElementById('jobRole').value,
        totalWorkingYears: document.getElementById('totalWorkingYears').value,
        yearsAtCompany: document.getElementById('yearsAtCompany').value,
        overTime: document.getElementById('overTime').value,
        businessTravel: document.getElementById('businessTravel').value
    };
    
    try {
        // Preprocess the input
        const processedInput = preprocessInput(inputData);
        
        // Convert to tensor
        const inputTensor = tf.tensor2d([processedInput]);
        
        // Make prediction
        const prediction = model.predict(inputTensor);
        const probability = (await prediction.data())[0];
        
        // Display result
        displayResult(probability);
        
        // Clean up
        inputTensor.dispose();
        prediction.dispose();
    } catch (error) {
        console.error("Prediction error:", error);
        alert("An error occurred during prediction. Check console for details.");
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