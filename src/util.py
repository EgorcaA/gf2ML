

def findExtrema(inputArray):
   arrayLength = len(inputArray)
   outputCount = 0
   for k in range(1, arrayLength - 1):
      outputCount += (inputArray[k] > inputArray[k - 1] and inputArray[k] > inputArray[k + 1])
      outputCount += (inputArray[k] < inputArray[k - 1] and inputArray[k] < inputArray[k + 1])
   return outputCount