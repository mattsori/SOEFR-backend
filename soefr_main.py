import asyncio
from websocket_server import start_websocket_server
from transcribeshort import start_transcribe_short
from transcribelong import start_transcribe_long

async def main():
    ws_task = asyncio.create_task(start_websocket_server())
    ts_task = asyncio.create_task(asyncio.to_thread(start_transcribe_short))
    tl_task = asyncio.create_task(asyncio.to_thread(start_transcribe_long))
    
    await asyncio.gather(ws_task, ts_task, tl_task)

if __name__ == "__main__":
    asyncio.run(main())