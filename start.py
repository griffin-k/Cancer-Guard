import asyncio
import subprocess
import os

def run_django():
   
    os.chdir('Guard')  
    subprocess.run(["python3", "manage.py", "runserver"])

async def run_fastapi():
    os.chdir('FastApi') 
    process = await asyncio.create_subprocess_exec(
        "uvicorn", "bot:app", "--reload", "--port", "8081", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await process.wait()

async def run_servers():

    loop = asyncio.get_event_loop()


    loop.run_in_executor(None, run_django)
    

    await run_fastapi()

if __name__ == "__main__":
    asyncio.run(run_servers())
