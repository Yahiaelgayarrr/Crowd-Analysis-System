from dataclasses import dataclass,field
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np



@dataclass
class Track:

    track_id:int

    cx:float
    cy:float

    vx:float=0
    vy:float=0

    age:int=0
    hits:int=1

    confirmed:bool=False

    history:list=field(
        default_factory=list
    )

    incubating:int=1




class PointTracker:


    def __init__(
        self,
        max_distance=40,
        max_age=90,
        min_hits=8,
        alpha=0.7
    ):

        self.max_distance=max_distance
        self.max_age=max_age
        self.min_hits=min_hits
        self.alpha=alpha

        self.tracks=[]

        self.next_id=1

        self.free_ids=set()



    def update(
        self,
        points,
        frame_num
    ):


        if len(points)==0:

            for t in self.tracks:

                t.age+=1

            return []



        predictions=[]

        for t in self.tracks:

            if t.age<5:

                px=t.cx+t.vx
                py=t.cy+t.vy

            else:

                px=t.cx
                py=t.cy

            predictions.append(
                [px,py]
            )


        if len(predictions)==0:

            predictions=np.empty(
                (0,2)
            )


        points_arr=np.array(points)


        if len(predictions):

            cost=cdist(
                predictions,
                points_arr
            )

            rows,cols=linear_sum_assignment(
                cost
            )

        else:

            rows=[]
            cols=[]



        matched=set()


        for r,c in zip(
            rows,
            cols
        ):

            if cost[r,c] > self.max_distance:

                continue

            t=self.tracks[r]

            oldx=t.cx
            oldy=t.cy

            nx,ny=points[c]


            t.vx=(
                self.alpha*
                (nx-oldx)
                +
                (1-self.alpha)*
                t.vx
            )


            t.vy=(
                self.alpha*
                (ny-oldy)
                +
                (1-self.alpha)*
                t.vy
            )

            t.cx=nx
            t.cy=ny

            t.age=0

            t.hits+=1

            t.incubating+=1


            if (
                t.incubating>=8
            ):

                t.confirmed=True


            t.history.append(
                (nx,ny)
            )

            t.history=t.history[-25:]

            matched.add(c)



        for i,p in enumerate(points):

            if i in matched:

                continue


            if len(self.free_ids):

                nid=min(
                    self.free_ids
                )

                self.free_ids.remove(
                    nid
                )

            else:

                nid=self.next_id

                self.next_id+=1


            self.tracks.append(

                Track(
                    nid,
                    p[0],
                    p[1]
                )

            )



        dead=[]

        for t in self.tracks:

            t.age+=1

            if t.age>self.max_age:

                dead.append(t)


        for t in dead:

            self.free_ids.add(
                t.track_id
            )

            self.tracks.remove(
                t)


        out=[]

        for t in self.tracks:

            if (
                t.confirmed
                and
                t.age<10
            ):

                out.append({

                    "track_id":
                    t.track_id,

                    "cx":
                    t.cx,

                    "cy":
                    t.cy

                })


        return out