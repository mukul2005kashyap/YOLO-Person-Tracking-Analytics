import numpy as np
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object ID to
        # its centroid and number of consecutive frames it has been marked as "disappeared", respectively
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given object is allowed to be
        # marked as "disappeared" before deregulation
        self.maxDisappeared = maxDisappeared
        
        # maximum distance between centroids to associate an object
        self.maxDistance = maxDistance

    def register(self, centroid, box):
        # when registering an object we use the next available object ID
        self.objects[self.nextObjectID] = {"centroid": centroid, "box": box}
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive frames where a 
                # given object has been marked as missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])

        # otherwise, are are currently tracking objects so we need to try to match the
        # input centroids to existing object centroids
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [obj["centroid"] for obj in self.objects.values()]

            # compute the distance between each pair of object centroids and input centroids
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)

            # find the smallest value in each row and then sort the row indexes based on their minimum values
            rows = D.min(axis=1).argsort()

            # perform a similar process on the columns by finding the smallest value in each
            # column and then sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # keep track of which of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than the max distance, do not associate
                if D[row, col] > self.maxDistance:
                    continue

                # capture the object ID, set its new centroid/box, and reset disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = {"centroid": inputCentroids[col], "box": rects[col]}
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is equal or greater than the number of input centroids
            # we need to check and see if some of these objects have potentially disappeared
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # otherwise, if the number of input centroids is greater than the number of existing object centroids
            # register each new input centroid
            for col in unusedCols:
                self.register(inputCentroids[col], rects[col])

        # return the set of trackable objects
        return self.objects
