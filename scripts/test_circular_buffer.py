import attr
import asyncio
import struct
import fire
import os, time
import numpy as np

from typing                        import Optional, List
from posix_ipc                     import Semaphore, ExistentialError, unlink_semaphore, unlink_shared_memory
from multiprocessing.shared_memory import SharedMemory

@attr.s
class Lock:
    name:   str                       = attr.ib()
    create: bool                      = attr.ib(default=False)
    loop:   asyncio.AbstractEventLoop = attr.ib(factory=asyncio.get_event_loop)
    sem:    Semaphore                 = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        if self.create:
            try:
                unlink_semaphore(self.name)
            except ExistentialError:
                pass

        flags = 0 if not self.create else os.O_CREAT | os.O_EXCL
        self.sem = Semaphore(self.name, flags=flags, initial_value=1)

    def __del__(self) -> None:
        # Creating the semaphore could have failed.
        if hasattr(self, 'sem'):
            if self.create:
                self.sem.unlink()

            self.sem.close()

    async def acquire(self) -> None:
        await self.loop.run_in_executor(None, lambda: self.sem.acquire())

    def release(self) -> None:
        self.sem.release()

    async def __aenter__(self) -> None:
        await self.loop.run_in_executor(None, lambda: self.sem.acquire())

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.sem.release()

@attr.s
class Barrier:
    name:       str                       = attr.ib()
    worker_id:  int                       = attr.ib()
    n_workers:  int                       = attr.ib()
    create:     bool                      = attr.ib(default=False)
    loop:       asyncio.AbstractEventLoop = attr.ib(factory=asyncio.get_event_loop)
    semaphores: List[Semaphore]           = attr.ib(init=False, default=[])

    def __attrs_post_init__(self) -> None:
        for i in range(self.n_workers):
            if self.create:
                try:
                    unlink_semaphore(self.name + f'_{i}')
                except ExistentialError:
                    pass

            flags = 0 if not self.create else os.O_CREAT | os.O_EXCL
            sem = Semaphore(self.name + f'_{i}', flags=flags, initial_value=0)
            self.semaphores.append(sem)

    def __del__(self) -> None:
        for sem in self.semaphores:
            if self.create:
                sem.unlink()

            try:
                sem.close()
            except ExistentialError:
                pass

    async def wait(self) -> None:
        for _ in range(self.n_workers - 1):
            self.semaphores[self.worker_id].release()

        tasks = [self.loop.run_in_executor(None, lambda i=i: self.semaphores[i].acquire()) for i in range(self.n_workers) \
            if i != self.worker_id]
        await asyncio.gather(*tasks)

class CircularBuffer:
    """
    Circular buffer intended for use with one reader process and one writer process. Multiple readers and writers are
    currently unsupported. The implementation is based on the one described here:
    https://ferrous-systems.com/blog/lock-free-ring-buffer/

    If the reads and writes for `read_off`, `write_off`, and `watermark_off` were guaranteed to be atomic with
    sequential consistency memory ordering, then we would not need locks.
    """

    """
    Memory reserved at the end of the buffer for the read offset, write offset, and watermark offset. We reserve 64
    bytes for each variable even though only 8 are strictly needed, so that they reside on separate cache lines. This
    eliminates unnecessary cache invalidation due to ping-ponging between CPU cores.

    Notes on variables:
      - `read_off` is the index into the buffer of the next byte to read. This byte might be invalid because it hasn't
        been written to yet.
      - `write_off` is the index into the buffer at which to write the next byte. This byte should be invalid.
      - When `write_off < read_off`, `watermark_off` is the index of the last valid byte in the buffer.
    """
    reserved_bytes: int = 64 + 64 + 64

    def __init__(self, name: str, size: int, create: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if create:
            try:
                unlink_shared_memory(name)
            except ExistentialError:
                pass

        self.create = create
        self.size   = size

        self.mem   = SharedMemory(name=name, size=self.size + self.reserved_bytes, create=self.create)
        self.loop  = loop if loop is not None else asyncio.get_event_loop()
        self.locks = [Lock(name + f'_lock_{i}', create=self.create, loop=self.loop) for i in range(3)]

    def __del__(self) -> None:
        # Creating the shared memory could have failed.
        if hasattr(self, 'mem'):
            if self.create:
                self.mem.unlink()

            self.mem.close()

    def __len__(self) -> int:
        return self.size

    def _get_read_off_unsafe(self) -> int:
        return struct.unpack_from('@Q', self.mem.buf, self.size)[0]

    async def _get_read_off(self) -> int:
        async with self.locks[0]:
            return self._get_read_off_unsafe()

    async def _set_read_off(self, off: int) -> None:
        assert off <= len(self) - 1

        async with self.locks[0]:
            struct.pack_into('@Q', self.mem.buf, self.size, off)

    def _get_write_off_unsafe(self) -> int:
        return struct.unpack_from('@Q', self.mem.buf, 64 + self.size)[0]

    async def _get_write_off(self) -> int:
        async with self.locks[1]:
            return self._get_write_off_unsafe()

    async def _set_write_off(self, off: int) -> None:
        assert off <= len(self) - 1

        async with self.locks[1]:
            struct.pack_into('@Q', self.mem.buf, 64 + self.size, off)

    def _get_watermark_off_unsafe(self) -> int:
        return struct.unpack_from('@Q', self.mem.buf, 2 * 64 + self.size)[0]

    async def _get_watermark_off(self) -> int:
        async with self.locks[2]:
            return self._get_watermark_off_unsafe()

    async def _set_watermark_off(self, off: int) -> None:
        assert off <= len(self) - 1

        async with self.locks[2]:
            struct.pack_into('@Q', self.mem.buf, 2 * 64 + self.size, off)

    async def read(self, n: int) -> Optional[bytes]:
        """
        Reads exactly `n` bytes from the buffer if possible, or else returns `None`.
        """

        read_off = self._get_read_off_unsafe()
        write_off = await self._get_write_off()
        watermark_off = await self._get_watermark_off()

        if write_off < read_off and read_off > watermark_off:
            read_off = 0

        if write_off >= read_off:
            if write_off - read_off < n:
                return None

            data = bytes(self.mem.buf[read_off : read_off + n])
            await self._set_read_off(read_off + n)
            return data
        else:
            assert watermark_off >= read_off

            """
            The total amount of data available to read is equal to the amount of data left at the end of the buffer (up
            to and including the watermark offset) and the data at the beginning.
            """
            if watermark_off - read_off + 1 + write_off < n:
                return None

            if n <= watermark_off - read_off + 1:
                data = bytes(self.mem.buf[read_off : read_off + n])
                await self._set_read_off(read_off + n if read_off + n < len(self) else 0)
                return data
            else:
                data = bytes(self.mem.buf[read_off : watermark_off + 1])
                n_rem = n - (watermark_off - read_off + 1)
                assert n_rem <= write_off

                data += self.mem.buf[:n_rem]
                await self._set_read_off(n_rem)
                return data

    async def write(self, data: bytes) -> bool:
        """
        Writes all the bytes in `data` to the buffer if possible and returns `True`, or else does nothing and returns
        `False`.
        """

        write_off = self._get_write_off_unsafe()
        watermark_off = self._get_watermark_off_unsafe()
        read_off = await self._get_read_off()

        if write_off >= read_off:
            if len(data) < len(self) - write_off:
                self.mem.buf[write_off : write_off + len(data)] = data # type: ignore
                await self._set_write_off(write_off + len(data))
                return True
            elif len(data) == len(self) - write_off:
                if read_off != 0:
                    self.mem.buf[write_off : write_off + len(data)] = data # type: ignore
                    await self._set_watermark_off(len(self) - 1)
                    await self._set_write_off(0)
                    return True
                else:
                    return False
            else:
                if len(data) < read_off:
                    """
                    We need to disallow `write_off == read_off` when `write_off < read_off` to avoid ambiguity.
                    """
                    assert write_off >= 1
                    self.mem.buf[:len(data)] = data # type: ignore
                    await self._set_watermark_off(write_off - 1)
                    await self._set_write_off(len(data))
                    return True
                else:
                    return False
        else:
            if len(data) < read_off - write_off:
                """
                We need to disallow `write_off == read_off` when `write_off < read_off` to avoid ambiguity.
                """
                self.mem.buf[write_off : write_off + len(data)] = data # type: ignore
                await self._set_write_off(write_off + len(data))
                return True
            else:
                return False

def test_1(role: str) -> None:
    """
    Tests reading and writing when wraparound does not occur.
    """

    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 16, True, loop)

        for i in range(8):
            await buf.write(str(i).encode('utf-8'))

        await asyncio.sleep(10000)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 16, False, loop)

        xs = [(await buf.read(1)).decode('utf-8') for _ in range(8)] # type: ignore
        assert xs == [str(x) for x in range(8)]

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_2(role: str) -> None:
    """
    Tests that reading fails when there are not enough bytes.
    """

    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 16, True, loop)

        for i in range(8):
            await buf.write(str(i).encode('utf-8'))

        await asyncio.sleep(10000)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 16, False, loop)

        xs = [await buf.read(1) for _ in range(9)]
        assert xs[-1] == None

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_3(role: str) -> None:
    """
    Tests that writing fails when there isn't enough room at the end of the buffer and nothing has been read.
    """

    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 16, True, loop)
        xs = [await buf.write(struct.pack('<B', i)) for i in range(17)]

        """
        Illustration for why the last two writes should fail. Suppose the buffer length is 3, and nothing has been read,
        so the read offset remains initialized to 0. After two writes, the write offset will be above the unwritten byte
        denoted by 'X'. We cannot write anything byte, because if we did, we would need to set the write offset to 0,
        which would violate the invariant for the two cases involving the watermark.

            v   v
            0 1 X
        """
        assert xs == 15 * [True] + 2 * [False]
        await asyncio.sleep(10000)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 16, False, loop)
        xs = [struct.unpack('<B', await buf.read(1))[0] for _ in range(15)] # type: ignore
        assert xs == list(range(15))

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_barrier(role: str) -> None:
    """
    Basic test to check that the IPC barrier works as expected.
    """

    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        barrier = Barrier('test', 0, 2, True)
        print("Waiting on barrier.")
        await barrier.wait()
        print("Past barrier.")

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        barrier = Barrier('test', 1, 2, False)
        print("Waiting on barrier.")
        await barrier.wait()
        print("Past barrier.")

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_4(role: str) -> None:
    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 9, True, loop)
        barrier = Barrier('barrier', 0, 2, True)

        await buf.write('0000'.encode('utf-8'))
        await buf.write('1111'.encode('utf-8'))
        await barrier.wait()

        await barrier.wait()
        # Tests that writer correctly jumps back to the start of the buffer
        await buf.write('2222'.encode('utf-8'))
        await barrier.wait()

        await asyncio.sleep(10000)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 9, False, loop)
        barrier = Barrier('barrier', 1, 2, False)

        await barrier.wait()
        print((await buf.read(4)).decode('utf-8'))
        print((await buf.read(4)).decode('utf-8'))
        await barrier.wait()

        await barrier.wait()
        print((await buf.read(4)).decode('utf-8'))

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_5(role: str) -> None:
    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 9, True, loop)
        barrier = Barrier('barrier', 0, 2, True)

        print(await buf.write('0000'.encode('utf-8')))
        print(await buf.write('11'.encode('utf-8')))
        await barrier.wait()

        await barrier.wait()
        print(await buf.write('11'.encode('utf-8')))
        # Tests that writer functions correctly when write_off < read_off
        print(await buf.write('2222'.encode('utf-8')))
        await barrier.wait()

        await asyncio.sleep(10000)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 9, False, loop)
        barrier = Barrier('barrier', 1, 2, False)

        await barrier.wait()
        print((await buf.read(6)).decode('utf-8'))
        await barrier.wait()

        await barrier.wait()
        print((await buf.read(2)).decode('utf-8'))
        print((await buf.read(4)).decode('utf-8'))

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_6(role: str) -> None:
    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 8, True, loop)
        barrier = Barrier('barrier', 0, 2, True)

        print(await buf.write('0000'.encode('utf-8')))
        await barrier.wait()

        await barrier.wait()
        print(await buf.write('1111'.encode('utf-8')))
        print(await buf.write('22'.encode('utf-8')))
        await barrier.wait()

        await asyncio.sleep(10000)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 8, False, loop)
        barrier = Barrier('barrier', 1, 2, False)

        await barrier.wait()
        print((await buf.read(3)).decode('utf-8'))
        await barrier.wait()

        await barrier.wait()
        # Test that reader correctly reads to end of buffer when `write_off < read_off`.
        print((await buf.read(5)).decode('utf-8'))
        print((await buf.read(2)).decode('utf-8'))

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_7(role: str) -> None:
    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 8, True, loop)
        barrier = Barrier('barrier', 0, 2, True)

        print(await buf.write('0000'.encode('utf-8')))
        await barrier.wait()

        await barrier.wait()
        print(await buf.write('1111'.encode('utf-8')))
        print(await buf.write('22'.encode('utf-8')))
        await barrier.wait()

        await asyncio.sleep(10000)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 8, False, loop)
        barrier = Barrier('barrier', 1, 2, False)

        await barrier.wait()
        print((await buf.read(3)).decode('utf-8'))
        await barrier.wait()

        await barrier.wait()
        # Test that reader correctly wraps around the end of buffer when `write_off < read_off`.
        print((await buf.read(7)).decode('utf-8'))

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

async def read_with_retry(buf: CircularBuffer, n: int) -> bytes:
    while True:
        r = await buf.read(n)

        if r is None:
            print("Read was blocked, will retry")
            # Sleeping for too long reduces the bandwidth with a concurrent reader and writer.
            await asyncio.sleep(0.1)
            continue

        return r

async def write_with_retry(buf: CircularBuffer, data: bytes) -> None:
    while True:
        r = await buf.write(data)

        if r:
            return

        print("Write was blocked, will retry")
        # Sleeping for too long reduces the bandwidth with a concurrent reader and writer.
        await asyncio.sleep(0.1)
        continue

def test_8(role: str) -> None:
    """
    Test concurrent reads and writes of random sizes.
    """

    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 128 * 2 ** 20, True, loop)
        n_written = 0

        while True:
            n = np.random.randint(64 * 2 ** 10, 128 * 2 ** 10)
            data = bytes(n)
            await write_with_retry(buf, data)
            n_written += n
            print(n_written)

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 128 * 2 ** 20, False, loop)
        n_read = 0

        while True:
            n = np.random.randint(64 * 2 ** 10, 128 * 2 ** 10)
            await read_with_retry(buf, n)
            n_read += n
            print(n_read)

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

def test_9(role: str) -> None:
    """
    Benchmarks reading and writing of 1 GiB. With a message size of 
    """

    async def run_producer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 256 * 2 ** 20, True, loop)
        n_written = 0
        t0 = time.time()

        while True:
            n = np.random.randint(128 * 2 ** 10, 256 * 2 ** 10)
            data = bytes(n)
            await write_with_retry(buf, data)
            n_written += n

            if n_written > 10 * 2 ** 30 + 256 * 2 ** 10:
                print(f"Wrote ~1 GiB in {time.time() - t0} sec.")
                return

    async def run_consumer(loop: asyncio.AbstractEventLoop) -> None:
        buf = CircularBuffer('test', 256 * 2 ** 20, False, loop)
        n_read = 0
        t0 = time.time()

        while True:
            n = 256 * 2 ** 10 #np.random.randint(128 * 2 ** 10, 256 * 2 ** 10)
            await read_with_retry(buf, n)
            n_read += n
            print(n_read)

            if n_read > 10 * 2 ** 30:
                print(f"Read ~1 GiB in {time.time() - t0} sec.")
                return

    loop = asyncio.get_event_loop()
    work_fn = {'producer': run_producer, 'consumer': run_consumer}[role]
    loop.run_until_complete(work_fn(loop))

if __name__ == '__main__':
    #fire.Fire(test_1)
    #fire.Fire(test_2)
    #fire.Fire(test_3)
    #fire.Fire(test_barrier)
    #fire.Fire(test_4)
    #fire.Fire(test_5)
    #fire.Fire(test_6)
    #fire.Fire(test_7)
    #fire.Fire(test_8)
    fire.Fire(test_9)
