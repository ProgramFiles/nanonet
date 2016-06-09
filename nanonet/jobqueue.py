from uuid import uuid4
import platform
from time import sleep
import os
from multiprocessing import Process
import Queue
from functools import partial

from myriad.components import MyriadServer
from myriad.managers import make_client

from nanonet.util import stderr_redirected

def times_two(x):
    return x, 2*x

def times_three(x):
    return x, 3*x

def multi_times_two(x):
    return [(y, 2*y) for y in x]

class JobQueue(object):

    def __init__(self, jobs, functors):
        """Watch a path and yield modified files

        """
        self.jobs = jobs
        self.functors = functors

    def _worker(self, function, take_n, timeout=0.5):
        sleep(2) # nasty, allows all workers to come up before iteration begins
        if take_n is None:
            self._singleton_worker(function, timeout=timeout)
        else:
            self._multi_worker(function, take_n, timeout=timeout)


    def _singleton_worker(self, function, timeout=0.5):
        manager = make_client(self.hostname, self.port, self.authkey)
        job_q = manager.get_job_q()
        job_q_closed = manager.q_closed()
        result_q = manager.get_result_q()

        while True:
            try: 
                job = job_q.get_nowait()
                result = function(job)
                result_q.put(result)
            except Queue.Empty:
                if job_q_closed._getvalue().value:
                    break
            sleep(timeout)

    def _multi_worker(self, function, take_n, timeout=0.5):
        manager = make_client(self.hostname, self.port, self.authkey)
        job_q = manager.get_job_q()
        job_q_closed = manager.q_closed()
        result_q = manager.get_result_q()

        while True:
            jobs = []
            try:
                for _ in xrange(take_n):
                    job = job_q.get_nowait()
                    jobs.append(job)
            except Queue.Empty:
                if job_q_closed._getvalue().value:
                    break
            else:
                for i, res in enumerate(function(jobs)):
                    result_q.put(res)
            sleep(timeout)
        if len(jobs) > 0:
            for i, res in enumerate(function(jobs)):
                result_q.put(res)

    def __iter__(self):
        self.start_server()
        workers = [Process(target=partial(self._worker, f[0], f[1])) for f in self.functors]

        try:
            for w in workers:
                w.start()

            for result in self.server.imap_unordered(self.jobs):
                yield result

            for w in workers:
                w.terminate()
        except KeyboardInterrupt:
            for w in workers:
                w.terminate()
            self.server.manager.join()
            self.server.manager.shutdown()

    def start_server(self, ports=(5000,6000)):
        self.authkey = str(uuid4())
        self.hostname = platform.uname()[1]

        server = None
        for port in xrange(*ports):
            try:
                with stderr_redirected(os.devnull):
                    server = MyriadServer(None, port, self.authkey)
            except EOFError:
                pass
            else:
                break
        if server is None:
            raise RuntimeError("Could not start myriad server.")

        self.server = server
        self.port = port
