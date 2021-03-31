import attr
import asyncio
import fire
import os, time

from posix_ipc import Semaphore, unlink_semaphore

@attr.s
class Lock:
    name: str       = attr.ib()
    sem:  Semaphore = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.sem = Semaphore(self.name, flags=os.O_CREAT, initial_value=1)

    def __del__(self) -> None:
        self.sem.close()

    def acquire(self) -> None:
        self.sem.acquire()

    def release(self) -> None:
        self.sem.release()

    def __enter__(self) -> None:
        self.sem.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.sem.release()

@attr.s
class ALock:
    name: str                       = attr.ib()
    loop: asyncio.AbstractEventLoop = attr.ib(factory=asyncio.get_event_loop)
    sem:  Semaphore                 = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.sem = Semaphore(self.name, flags=os.O_CREAT, initial_value=1)

    def __del__(self) -> None:
        self.sem.close()

    async def acquire(self) -> None:
        await self.loop.run_in_executor(None, lambda: self.sem.acquire())

    async def release(self) -> None:
        await self.loop.run_in_executor(None, lambda: self.sem.release())

    async def __aenter__(self) -> None:
        await self.loop.run_in_executor(None, lambda: self.sem.acquire())

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.loop.run_in_executor(None, lambda: self.sem.release())

def test_1(role: str) -> None:
    """
    The producer should be run before the consumer.
    """

    if role == 'producer':
        lock = Lock('test')

        with lock:
            print("Producer acquired lock.")
            time.sleep(5)

        print("Producer released lock.")
    elif role == 'consumer':
        lock = Lock('test')
        print("Consumer waiting to acquire lock.")

        with lock:
            print("Consumer acquired lock.")

        print("Consumer released lock.")

async def run_producer() -> None:
    lock = ALock('test')

    async with lock:
        print("Producer acquired lock.")
        time.sleep(5)

    print("Producer released lock.")

async def run_consumer() -> None:
    lock = ALock('test')
    print("Consumer waiting to acquire lock.")

    async with lock:
        print("Consumer acquired lock.")

    print("Consumer released lock.")

def test_2(role: str) -> None:
    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn())

if __name__ == '__main__':
    #fire.Fire(test_1)
    fire.Fire(test_2)
