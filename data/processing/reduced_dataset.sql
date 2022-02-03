-- Reduce dataset to last 7 years 
SELECT *
INTO [dbo].[Seattle_Real_Time_Fire_911_Calls_7_years]
FROM [EmergencyCallsDB].[dbo].[Seattle_Real_Time_Fire_911_Calls]
WHERE [Datetime] >= DATEADD(YEAR, -7, GETDATE())

