-- Reduce dataset to last 7 years 
SELECT 
    [Address],
    [Type],
    [Latitude],
    [Longitude],
    [Report_Location],
    [Incident_Number],
    CONVERT(DATE,[Datetime]) AS [Date],
    CONVERT(TIME(0),[Datetime]) AS [Time]
FROM [EmergencyCallsDB].[dbo].[Seattle_Real_Time_Fire_911_Calls_7_years]