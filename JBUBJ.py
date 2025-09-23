"use client";

import

{useState, useEffect}
from

"react";
import

{Card, CardContent, CardDescription, CardHeader, CardTitle}
from

"@/components/ui/card";
import

{Button}
from

"@/components/ui/button";
import

{Input}
from

"@/components/ui/input";
import

{Label}
from

"@/components/ui/label";
import

{Select, SelectContent, SelectItem, SelectTrigger, SelectValue}
from

"@/components/ui/select";
import

{LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell}
from

"recharts";
import

{Calendar, Clock, Shield, Users, Heart}
from

"lucide-react";

// Types
for our data
    type SensorData = {
        timestamp: string;
    temperature: number;
    humidity: number;
    soilMoisture: number;
    phLevel: number;
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    };

    type CropHealth = {
        overall: number;
    leafHealth: number;
    rootHealth: number;
    growthStage: string;
    };

    type PestRisk = {
        riskLevel: number;
    pestType: string;
    affectedArea: number;
    };

    type HistoricalRecord = {
        id: string;
    date: string;
    cropHealth: number;
    soilCondition: number;
    pestRisk: number;
    notes: string;
    };

    // Mock
    data
    generators
    const
    generateSensorData = (): SensorData = > ({
                                                 timestamp: new Date().toISOString(),
                                             temperature: 22 + Math.random() * 5,
    humidity: 60 + Math.random() * 20,
    soilMoisture: 40 + Math.random() * 30,
    phLevel: 6.5 + Math.random() * 1,
    nitrogen: 120 + Math.random() * 50,
    phosphorus: 80 + Math.random() * 30,
    potassium: 150 + Math.random() * 40,
    });

    const
    generateCropHealth = (): CropHealth = > ({
        overall: 85 + Math.random() * 10,
        leafHealth: 80 + Math.random() * 15,
        rootHealth: 90 + Math.random() * 8,
        growthStage: ["Vegetative", "Flowering", "Fruiting"][Math.floor(Math.random() * 3)],
    });

    const
    generatePestRisk = (): PestRisk = > ({
        riskLevel: Math.floor(Math.random() * 4),
        pestType: ["Aphids", "Caterpillars", "Spider Mites", "Fungus Gnats"][Math.floor(Math.random() * 4)],
        affectedArea: Math.random() * 15,
    });

    const
    generateHistoricalData = (): HistoricalRecord[] = > {
        const
    records: HistoricalRecord[] = [];
    for (let i = 30; i >= 0; i--) {
        const date = new Date();
    date.setDate(date.getDate() - i);
    records.push({
    id: `rec -${i}
    `,
    date: date.toISOString().split('T')[0],
    cropHealth: 75 + Math.random() * 20,
    soilCondition: 70 + Math.random() * 25,
    pestRisk: Math.floor(Math.random() * 4),
    notes: `Day ${30 - i}
    observation
    `,
    });
    }
    return records;
    };

    // Color
    palette
    for charts
    const COLORS =['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

    export default function AgriculturalDashboard() {
    const[sensorData, setSensorData] = useState < SensorData > (generateSensorData());
    const[cropHealth, setCropHealth] = useState < CropHealth > (generateCropHealth());
    const[pestRisk, setPestRisk] = useState < PestRisk > (generatePestRisk());
    const[historicalData, setHistoricalData] = useState < HistoricalRecord[] > (generateHistoricalData());
    const[newRecord, setNewRecord] = useState({
        cropHealth: 80,
        soilCondition: 75,
        pestRisk: 1,
        notes: ""
    });

    // Simulate
    real - time
    data
    updates
    useEffect(() = > {
        const
    interval = setInterval(() = > {
        setSensorData(generateSensorData());
    setCropHealth(generateCropHealth());
    setPestRisk(generatePestRisk());
    }, 10000); // Update
    every
    10
    seconds

    return () = > clearInterval(interval);
    }, []);

    // Prepare
    data
    for charts
    const healthData =[
    {name: 'Overall', value: cropHealth.overall},
    {name: 'Leaf Health', value: cropHealth.leafHealth},
    {name: 'Root Health', value: cropHealth.rootHealth},
    ];

    const
    riskData = [
        {name: 'Low', value: pestRisk.riskLevel == = 0 ? 100: 0},
    {name: 'Medium', value: pestRisk.riskLevel == = 1 ? 100: 0},
    {name: 'High', value: pestRisk.riskLevel == = 2 ? 100: 0},
    {name: 'Critical', value: pestRisk.riskLevel == = 3 ? 100: 0},
    ];

    const
    sensorHistory = historicalData.map(record= > ({
                                                      date: record.date,
                                                      cropHealth: record.cropHealth,
                                                      soilCondition: record.soilCondition,
                                                      pestRisk: record.pestRisk * 25, // Scale for visualization
    }));

    const
    handleAddRecord = () = > {
        const
    newRecordObj: HistoricalRecord = {
    id: `rec -${Date.now()}
    `,
    date: new
    Date().toISOString().split('T')[0],
    cropHealth: newRecord.cropHealth,
    soilCondition: newRecord.soilCondition,
    pestRisk: newRecord.pestRisk,
    notes: newRecord.notes,
    };

    setHistoricalData([...historicalData, newRecordObj]);
    setNewRecord({
        cropHealth: 80,
        soilCondition: 75,
        pestRisk: 1,
        notes: ""
    });
    };

    return (
        < div className="min-h-screen bg-gradient-to-b from-green-50 to-cyan-50 p-4 md:p-8" >
        < div className="max-w-7xl mx-auto" >
        {/ * Header * /}
        < header className="mb-8" >
        < div className="flex flex-col md:flex-row md:items-center md:justify-between" >
        < div >
        < h1 className="text-3xl font-bold text-green-800" > AgriVision Dashboard < / h1 >
        < p className="text-green-600" > AI-powered crop monitoring and analytics < / p >
        < / div >
        < div className="flex items-center space-x-4 mt-4 md:mt-0" >
        < div className="flex items-center text-green-700" >
        < Calendar className="mr-2 h-5 w-5" / >
        < span > {new Date().toLocaleDateString()} < / span >
                                                       < / div >
                                                           < div
    className = "flex items-center text-green-700" >
                < Clock
    className = "mr-2 h-5 w-5" / >
                < span > {new
    Date().toLocaleTimeString()} < / span >
                                     < / div >
                                         < / div >
                                             < / div >
                                                 < / header >

                                                     { / * Real - time
    Monitoring
    Section * /}
    < section
    className = "mb-12" >
                < h2
    className = "text-2xl font-semibold text-green-800 mb-6" > Real - time
    Monitoring < / h2 >

                   < div
    className = "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8" >
                { / * Sensor
    Data
    Cards * /}
    < Card
    className = "bg-white shadow-md" >
                < CardHeader
    className = "pb-2" >
                < CardTitle
    className = "text-lg flex items-center" >
                < Shield
    className = "mr-2 text-blue-500" / >
                Environmental
    Conditions
    < / CardTitle >
        < / CardHeader >
            < CardContent >
            < div
    className = "space-y-3" >
                < div
    className = "flex justify-between" >
                < span > Temperature < / span >
                                         < span
    className = "font-medium" > {sensorData.temperature.toFixed(1)}°C < / span >
                                                                          < / div >
                                                                              < div
    className = "flex justify-between" >
                < span > Humidity < / span >
                                      < span
    className = "font-medium" > {sensorData.humidity.toFixed(1)} % < / span >
                                                                       < / div >
                                                                           < div
    className = "flex justify-between" >
                < span > Soil
    Moisture < / span >
                 < span
    className = "font-medium" > {sensorData.soilMoisture.toFixed(1)} % < / span >
                                                                           < / div >
                                                                               < div
    className = "flex justify-between" >
                < span > pH
    Level < / span >
              < span
    className = "font-medium" > {sensorData.phLevel.toFixed(2)} < / span >
                                                                    < / div >
                                                                        < / div >
                                                                            < / CardContent >
                                                                                < / Card >

                                                                                    { / * Nutrient
    Levels * /}
    < Card
    className = "bg-white shadow-md" >
                < CardHeader
    className = "pb-2" >
                < CardTitle
    className = "text-lg" > Nutrient
    Levels < / CardTitle >
               < / CardHeader >
                   < CardContent >
                   < div
    className = "space-y-3" >
                < div
    className = "flex justify-between" >
                < span > Nitrogen(N) < / span >
                                         < span
    className = "font-medium" > {sensorData.nitrogen.toFixed(1)}
    ppm < / span >
            < / div >
                < div
    className = "flex justify-between" >
                < span > Phosphorus(P) < / span >
                                           < span
    className = "font-medium" > {sensorData.phosphorus.toFixed(1)}
    ppm < / span >
            < / div >
                < div
    className = "flex justify-between" >
                < span > Potassium(K) < / span >
                                          < span
    className = "font-medium" > {sensorData.potassium.toFixed(1)}
    ppm < / span >
            < / div >
                < / div >
                    < / CardContent >
                        < / Card >

                            { / * Crop
    Health * /}
    < Card
    className = "bg-white shadow-md" >
                < CardHeader
    className = "pb-2" >
                < CardTitle
    className = "text-lg flex items-center" >
                < Heart
    className = "mr-2 text-red-500" / >
                Crop
    Health
    < / CardTitle >
        < / CardHeader >
            < CardContent >
            < div
    className = "flex flex-col items-center" >
                < div
    className = "relative w-32 h-32" >
                < div
    className = "absolute inset-0 flex items-center justify-center" >
                < span
    className = "text-2xl font-bold text-green-700" > {cropHealth.overall.toFixed(0)} % < / span >
                                                                                            < / div >
                                                                                                < svg
    className = "w-full h-full"
    viewBox = "0 0 36 36" >
              < path
    d = "M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
    fill = "none"
    stroke = "#eee"
    strokeWidth = "3"
                  / >
                  < path
    d = "M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
    fill = "none"
    stroke = {cropHealth.overall > 80 ? "#4ade80": cropHealth.overall > 60 ? "#fbbf24": "#f87171"}
    strokeWidth = "3"
    strokeDasharray = {`${cropHealth.overall}, 100
    `}
    / >
    < / svg >
        < / div >
            < p
    className = "mt-2 text-sm text-gray-600" > Growth
    Stage: {cropHealth.growthStage} < / p >
                                        < / div >
                                            < / CardContent >
                                                < / Card >

                                                    { / * Pest
    Risk * /}
    < Card
    className = "bg-white shadow-md" >
                < CardHeader
    className = "pb-2" >
                < CardTitle
    className = "text-lg" > Pest
    Risk
    Assessment < / CardTitle >
                   < / CardHeader >
                       < CardContent >
                       < div
    className = "flex flex-col items-center" >
                < div
    className = "text-center mb-3" >
                < span
    className = {`text - xl
    font - bold ${
        pestRisk.riskLevel == = 0 ? "text-green-600":
    pestRisk.riskLevel == = 1 ? "text-yellow-500":
    pestRisk.riskLevel == = 2 ? "text-orange-500": "text-red-600"
    }`} >
    {pestRisk.riskLevel == = 0 ? "Low":
    pestRisk.riskLevel == = 1 ? "Medium":
    pestRisk.riskLevel == = 2 ? "High": "Critical"}
    < / span >
        < p
    className = "text-sm text-gray-600 mt-1" > Risk
    Level < / p >
              < / div >
                  < p
    className = "text-sm text-center" >
                Detected: < span
    className = "font-medium" > {pestRisk.pestType} < / span >
                                                        < / p >
                                                            < p
    className = "text-sm text-center" >
                Affected
    Area: < span
    className = "font-medium" > {pestRisk.affectedArea.toFixed(1)} % < / span >
                                                                         < / p >
                                                                             < / div >
                                                                                 < / CardContent >
                                                                                     < / Card >
                                                                                         < / div >

                                                                                             { / * Charts
    Section * /}
    < div
    className = "grid grid-cols-1 lg:grid-cols-2 gap-6" >
                { / * Health
    Metrics
    Chart * /}
    < Card
    className = "bg-white shadow-md" >
                < CardHeader >
                < CardTitle > Crop
    Health
    Metrics < / CardTitle >
                < CardDescription > Current
    health
    indicators < / CardDescription >
                   < / CardHeader >
                       < CardContent >
                       < ResponsiveContainer
    width = "100%"
    height = {300} >
             < BarChart
    data = {healthData} >
           < CartesianGrid
    strokeDasharray = "3 3" / >
                      < XAxis
    dataKey = "name" / >
              < YAxis
    domain = {[0, 100]} / >
             < Tooltip / >
             < Bar
    dataKey = "value" >
              {healthData.map((entry, index) = > (
        < Cell key={`cell-${index}`} fill={COLORS[index %COLORS.length]} / >
    ))}
    < / Bar >
        < / BarChart >
            < / ResponsiveContainer >
                < / CardContent >
                    < / Card >

                        { / * Pest
    Risk
    Chart * /}
    < Card
    className = "bg-white shadow-md" >
                < CardHeader >
                < CardTitle > Pest
    Risk
    Distribution < / CardTitle >
                     < CardDescription > Current
    risk
    assessment < / CardDescription >
                   < / CardHeader >
                       < CardContent >
                       < ResponsiveContainer
    width = "100%"
    height = {300} >
             < PieChart >
             < Pie
    data = {riskData}
    cx = "50%"
    cy = "50%"
    labelLine = {false}
    outerRadius = {80}
    fill = "#8884d8"
    dataKey = "value"
    label = {({name, percent}) = > `${name}: ${(percent * 100).toFixed(0)} % `}
    >
    {riskData.map((entry, index) = > (
        < Cell key={`cell-${index}`} fill={
        entry.name == = "Low" ? "#4ade80":
        entry.name == = "Medium" ? "#fbbf24":
        entry.name == = "High" ? "#f97316": "#ef4444"
        } / >
    ))}
    < / Pie >
        < Tooltip / >
        < / PieChart >
            < / ResponsiveContainer >
                < / CardContent >
                    < / Card >
                        < / div >
                            < / section >

                                { / * Historical
    Data
    Section * /}
    < section >
      < h2
    className = "text-2xl font-semibold text-green-800 mb-6" > Historical
    Data & Records < / h2 >

                       < div
    className = "grid grid-cols-1 lg:grid-cols-3 gap-6" >
                { / * Add
    New
    Record
    Form * /}
    < Card
    className = "bg-white shadow-md lg:col-span-1" >
                < CardHeader >
                < CardTitle > Add
    New
    Record < / CardTitle >
               < CardDescription > Enter
    historical
    data < / CardDescription >
             < / CardHeader >
                 < CardContent >
                 < div
    className = "space-y-4" >
                < div >
                < Label
    htmlFor = "cropHealth" > Crop
    Health( %) < / Label >
                   < Input
    id = "cropHealth"
    type = "number"
    min = "0"
    max = "100"
    value = {newRecord.cropHealth}
    onChange = {(e) = > setNewRecord({...
    newRecord, cropHealth: Number(e.target.value)})}
    / >
    < / div >

        < div >
        < Label
    htmlFor = "soilCondition" > Soil
    Condition( %) < / Label >
                      < Input
    id = "soilCondition"
    type = "number"
    min = "0"
    max = "100"
    value = {newRecord.soilCondition}
    onChange = {(e) = > setNewRecord({...
    newRecord, soilCondition: Number(e.target.value)})}
    / >
    < / div >

        < div >
        < Label
    htmlFor = "pestRisk" > Pest
    Risk < / Label >
             < Select
    value = {newRecord.pestRisk.toString()}
    onValueChange = {(value) = > setNewRecord({...
    newRecord, pestRisk: Number(value)})}
    >
    < SelectTrigger >
      < SelectValue
    placeholder = "Select risk level" / >
                  < / SelectTrigger >
                      < SelectContent >
                      < SelectItem
    value = "0" > Low < / SelectItem >
                          < SelectItem
    value = "1" > Medium < / SelectItem >
                             < SelectItem
    value = "2" > High < / SelectItem >
                           < SelectItem
    value = "3" > Critical < / SelectItem >
                               < / SelectContent >
                                   < / Select >
                                       < / div >

                                           < div >
                                           < Label
    htmlFor = "notes" > Notes < / Label >
                                  < Input
    id = "notes"
    value = {newRecord.notes}
    onChange = {(e) = > setNewRecord({...
    newRecord, notes: e.target.value})}
    placeholder = "Observations..."
                  / >
                  < / div >

                      < Button
    className = "w-full bg-green-600 hover:bg-green-700"
    onClick = {handleAddRecord} >
              Add
    Record
    < / Button >
        < / div >
            < / CardContent >
                < / Card >

                    { / * Historical
    Trends
    Chart * /}
    < Card
    className = "bg-white shadow-md lg:col-span-2" >
                < CardHeader >
                < CardTitle > Historical
    Trends < / CardTitle >
               < CardDescription > 30 - day
    overview
    of
    crop
    health and conditions < / CardDescription >
                              < / CardHeader >
                                  < CardContent >
                                  < ResponsiveContainer
    width = "100%"
    height = {400} >
             < LineChart
    data = {sensorHistory} >
           < CartesianGrid
    strokeDasharray = "3 3" / >
                      < XAxis
    dataKey = "date" / >
              < YAxis / >
              < Tooltip / >
              < Legend / >
              < Line
    type = "monotone"
    dataKey = "cropHealth"
    stroke = "#10b981"
    strokeWidth = {2}
    activeDot = {{r: 8}} / >
                < Line
    type = "monotone"
    dataKey = "soilCondition"
    stroke = "#3b82f6"
    strokeWidth = {2} / >
                  < Line
    type = "monotone"
    dataKey = "pestRisk"
    stroke = "#ef4444"
    strokeWidth = {2} / >
                  < / LineChart >
                      < / ResponsiveContainer >
                          < / CardContent >
                              < / Card >
                                  < / div >
                                      < / section >

                                          { / * Footer * /}
    < footer
    className = "mt-12 pt-6 border-t border-green-200 text-center text-green-700" >
                < p > AgriVision
    Dashboard • AI - powered
    agricultural
    monitoring
    system < / p >
               < p
    className = "text-sm mt-2" > Data
    updates
    every
    10
    seconds • Last
    updated: {new
    Date().toLocaleTimeString()} < / p >
                                     < / footer >
                                         < / div >
                                             < / div >
    );
    }
    Share
    Refresh
    Copy