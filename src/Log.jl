
export SetLogLevel, SetLogToFileLevel
export LogLow, Log, LogHigh, LogMedium
export SetLogFile, SetPriorityLogFile

LogLevel = 0
LogToFileLevel = 2

LogFile = "Log.txt"
PriorityLogFile = "Summary.txt"

open(PriorityLogFile, "w") do file
end
open(LogFile, "w") do file
end

function Log(text)
    open(LogFile, "a") do file
        write(file, text * "\n")
    end
    open(PriorityLogFile, "a") do file
        write(file, text * "\n")
    end
    println(text)
end

function SetLogLevel(level::Int64)
    global LogLevel = level
end

function SetLogToFileLevel(level::Int64)
    global LogToFileLevel = level
end

function SetLogFile(path)
    global LogFile = path
    open(LogFile, "w") do file
    end
end

function SetPriorityLogFile(path)
    global PriorityLogFile = path
    open(PriorityLogFile, "w") do file
    end
end


function LogLow(info)
    if LogLevel >= 3
        println(info)
    end
    if LogToFileLevel >= 3
        open(LogFile, "a") do file
            write(file, info * "\n")
        end
    end
end

function LogMedium(info)
    if LogLevel >= 2
        println(info)
    end
    if LogToFileLevel >= 2
        open(LogFile, "a") do file
            write(file, info * "\n")
        end
    end
end

function LogHigh(info)
    if LogLevel >= 1
        println(info)
    end
    if LogToFileLevel >= 1
        open(LogFile, "a") do file
            write(file, info * "\n")
        end
    end
end
