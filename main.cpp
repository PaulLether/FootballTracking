/**
    Paul Lether Football Tracking program. Uses a video input to track football
    players distances during a football match. Carmera calibration is estimated
    and therefore has to be hardcoded in. To use new videos you need to updated
    these hardcoded values.

    Must use opencv with 3.0.1 and with the SIFT package installed

    Data set = Camera setting 2, from http://home.ifi.uio.no/paalh/dataset/alfheim/
    */

#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct Player {
  int id;
  Point2f tl;
  Point2f br;
  Point2f midpoint;
  double distanceMoved;
  Scalar colour;
  vector<Mat> SIFTFeatures;
  Scalar meanColourRed;
} ;

/**
    Returns the video input that will be used

    @param argv[] The command line input
    @return The video that needs to be opened
*/
VideoCapture setUpInput(char * argv[],  VideoCapture cap)
{
  /* Create a VideoCapture object and open the input file
    If the input is taken from the camera, pass 0 instead
    of the video file name. Number 1 works with current set up
  */
  if(strcmp(argv[1],"1") == 0 )
    cap.open("output.mp4");
  return cap;
}

/**
    Euclidean distance between two points using

    @param p point 1
    @param q point 2
    @return The distance between the two points
*/
float distance(Point2f p, Point2f q)
{
  Point diff = p - q;
  return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

/**
    hard coded points of the curve line of the top and bottom sides
    of the pitch

    @param objectPoints array to hold the points created
    @return objectPoints array to hold the points created
*/
vector<vector< Point3f >> curved_Lines(vector<vector< Point3f >> objectPoints)
{
  // code from stackoverflow
  // https://stackoverflow.com/questions/26602981/
  //correct-barrel-distortion-in-opencv-manually-without
  //-chessboard-image

  // top line
  objectPoints[0].push_back(Point3f(1390,635,0));
  objectPoints[0].push_back(Point3f(1630,595,0));
  objectPoints[0].push_back(Point3f(1870,565,0));
  objectPoints[0].push_back(Point3f(2110,550,0));
  objectPoints[0].push_back(Point3f(2350,545,0));
  objectPoints[0].push_back(Point3f(2590,550,0));
  objectPoints[0].push_back(Point3f(2830,570,0));
  objectPoints[0].push_back(Point3f(3070,605,0));

  // bottom line
  objectPoints[0].push_back(Point3f(486,1445 ,0));
  objectPoints[0].push_back(Point3f(922,1590,0));
  objectPoints[0].push_back(Point3f(1350,1700,0));
  objectPoints[0].push_back(Point3f(1794,1765,0));  // middle line
  objectPoints[0].push_back(Point3f(2230,1790,0));
  objectPoints[0].push_back(Point3f(2666,1760,0));
  objectPoints[0].push_back(Point3f(3102,1685,0));
  objectPoints[0].push_back(Point3f(3538,1560,0));
  objectPoints[0].push_back(Point3f(3974,1400,0));

  return objectPoints;
}

/**
    hard coded points of what the straight line of the top and bottom sides
    of the pitch should be

    @param imagePoints array to hold the points created
    @return imagePoints array to hold the points created
*/
vector<vector< Point2f >> equation_of_Straight_Line(vector<vector< Point2f >> imagePoints)
{
  // top line
  imagePoints[0].push_back(Point2f(1390,673.53));
  imagePoints[0].push_back(Point2f(1630, 667.06));
  imagePoints[0].push_back(Point2f(1870,660.59));
  imagePoints[0].push_back(Point2f(2210,654.12));
  imagePoints[0].push_back(Point2f(2350,647.65));
  imagePoints[0].push_back(Point2f(2590,641.18));
  imagePoints[0].push_back(Point2f(2830,634.71));
  imagePoints[0].push_back(Point2f(3070,628.24));

  // bottom line
  imagePoints[0].push_back(Point2f(486,1246.5));
  imagePoints[0].push_back(Point2f(922,1243));
  imagePoints[0].push_back(Point2f(1350,1239.56));
  imagePoints[0].push_back(Point2f(1794,1236));  // middle line
  imagePoints[0].push_back(Point2f(2230,1232.5));
  imagePoints[0].push_back(Point2f(2666,1229));
  imagePoints[0].push_back(Point2f(3102,1225.5));
  imagePoints[0].push_back(Point2f(3538,1222));
  imagePoints[0].push_back(Point2f(3974,1218.5));

  return imagePoints;
}

/**
    finds all the green pixels of a given image

    @param frame the image you wish to find the green pixels of
    @return imagePoints image with only green pixels
*/
Mat green_pixels_of_image(Mat *frame)
{
  Mat mask, greenOnly, hsvFrame;
  cvtColor(*frame, hsvFrame, CV_BGR2HSV);
  // finding the green area of a frame
  inRange(hsvFrame, Scalar(45, 130, 10), Scalar(66, 255, 255), mask);

  // if the pixel is green then keep its value in the orignal photo
  bitwise_and(*frame, *frame, greenOnly, mask);
  return greenOnly;
}

/**
    finds the largest contour of an image

    @param frame the image you wish to find the green pixels of
    @return imagePoints returns an image with with largest contour only
*/
Mat largest_contour(Mat *greenOnly)
{
  Mat maskNothing,  greyGreenOnly, croppedPitch;
  vector<vector<cv::Point>> greencontours; // Vector for storing contour
  Mat holeInPitch =  getStructuringElement( MORPH_RECT,
                  Size( 3, 3 ),
                  Point( 2, 2 ) );

  // creating a blank image
  cvtColor(*greenOnly, greyGreenOnly, CV_BGR2GRAY);
  inRange(*greenOnly, Scalar(0, 0, 1), Scalar(0, 0, 1), maskNothing);
  bitwise_and(*greenOnly, *greenOnly, croppedPitch, maskNothing);

  GaussianBlur(greyGreenOnly, greyGreenOnly, Size(5, 5), 0);

  for(int i = 0 ; i < 3 ; i++)
    erode( greyGreenOnly, greyGreenOnly, holeInPitch );

  for(int i = 0 ; i < 3 ; i++)
    dilate(greyGreenOnly, greyGreenOnly, holeInPitch);

  findContours( greyGreenOnly, greencontours, CV_RETR_EXTERNAL,
    CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
  float biggest_Area= 0.0;

  for( size_t i = 0; i< greencontours.size(); i++ ) // iterate through each contour.
  {
    if(contourArea(greencontours[i]) > biggest_Area)
    {
      biggest_Area = contourArea(greencontours[i]);
    }
  }

  for( size_t i = 0; i< greencontours.size(); i++ ) // iterate through each contour
  {
    float area = contourArea( greencontours[i] );  //  Find the area of contour

    if( area == biggest_Area)
    {
      drawContours( croppedPitch, greencontours,i,
        Scalar( 0, 255, 0 ), -1); // Draw the largest contour using previously stored index.
    }
  }
  return croppedPitch;
}

/**
    finds the lines around the pitch

    @param frame the correct frame of the pitch
    @param croppedPitch an image of the pitch area (ish) only
    @return an image with only the lines of the pitch
*/
Mat find_Lines_Around_Pitch(Mat *frame, Mat *croppedPitch)
{
  Mat edge, draw, mask, linesOfPitch;

  // getting the area(ish) of the pitch only
  bitwise_and(*frame, *croppedPitch, *croppedPitch);

  // finding the lines of the pitch itself
  Canny(*croppedPitch, edge, 160, 240, 3);
  edge.convertTo(draw, CV_8U);
  vector<Vec4i> lines;
  HoughLinesP(draw, lines, 1, CV_PI/180, 40, 20, 150);

  Mat frameWithLines = frame->clone();
  // drawing on the lines of the pitch
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( frameWithLines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, CV_AA);
  }

  // masking the pitch so only the lines of the pitch appear
  inRange(frameWithLines, Scalar(0,0,245), Scalar(0,0,255), mask);
  bitwise_and(frameWithLines, frameWithLines, linesOfPitch, mask);
  return linesOfPitch;
}

/**
    finds the corner of the pitch given a image of of lines around the pitch.
    It does this by taking the parameter of the lines of the pitch and then
    finding the corners by using the haris corner detector. It then splits the
    image into a grid of 100 by 50 and counts the number of corners in each grid
    to find an average.

    The bottom left corner will always be the lowest x value
    and the right corner will be the hightest x value due to the perpective view.

    the top left is worked out by finding the first grid as that has a value,
    going from the top left value downwards after going through every column on
    this row. This value will normal be the top left of the pitch however; because
    the video image still has an barrel effect this is infact the middle. So a
    grid threshold is used is used where we find the highest and lowest x value
    within this y threshold.

    @param linesAroundPitch image with lines of the pitch
    @return pitchPoints corners of the pitch
*/
vector<Point2f> find_corners_Of_Pitch(Mat *linesAroundPitch)
{
  // variable for the corners of the pitch
  vector<Point2f> pitchPoints(4);

  // variable for harris corner
  int blockSize = 2;
  int apertureSize = 3;
  int thresh = 100;
  double k = 0.04;
  Mat src_gray;

  // variables for the grid
  int gridColNum = 100;
  int gridRowNum = 50;
  float pitchSquarex[gridColNum][gridRowNum];
  float pitchSquarey[gridColNum][gridRowNum];
  int pitchSquaretotal[gridColNum][gridRowNum];

  // harris corner detection
  cvtColor( *linesAroundPitch, src_gray, COLOR_BGR2GRAY );
  Mat dst = Mat::zeros( linesAroundPitch->size(), CV_32FC1 );
  cornerHarris( src_gray, dst, blockSize, apertureSize, k );
  Mat dst_norm, dst_norm_scaled;
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );

  for(int i = 0 ; i < gridColNum; i++)
  {
    for(int j = 0 ; j < gridRowNum; j++)
    {
      pitchSquarex[i][j] = 0;
      pitchSquarey[i][j] = 0;
      pitchSquaretotal[i][j] = 0;
    }
  }

  // finding out how big each square of the grid needs to be
  float widthSquareSize = linesAroundPitch->cols / gridColNum;
  float heightSquareSize =  linesAroundPitch->rows / gridRowNum;

  // finding the number of corners in each grid
  for( int i = 0; i < dst_norm.rows ; i++ )
  {
    for( int j = 0; j < dst_norm.cols; j++ )
    {
      if((int) dst_norm.at<float>(i,j) > thresh )
      {
        // finding what square the point is in
        int columnValue = j / widthSquareSize;
        int rowValue = i /  heightSquareSize ;

        pitchSquarex[columnValue][rowValue] = pitchSquarex[columnValue][rowValue] + j;
        pitchSquarey[columnValue][rowValue] = pitchSquarey[columnValue][rowValue] + i;
        pitchSquaretotal[columnValue][rowValue] = pitchSquaretotal[columnValue][rowValue] + 1;
      }
    }
  }

  // finding the average in each grid square
  for(int i = 0 ; i < gridColNum; i++)
  {
    for(int j = 0 ; j < gridRowNum; j++)
    {
      if(pitchSquaretotal[i][j] > 0)
      {
        pitchSquarex[i][j] = pitchSquarex[i][j] / pitchSquaretotal[i][j];
        pitchSquarey[i][j] = pitchSquarey[i][j] / pitchSquaretotal[i][j];
      }
    }
  }

  // finding all the corners of the pitch
  bool firstValue = true;
  float firstValueGridY;
  float thresholdHeight = 2;
  for(int j = 0 ; j < gridRowNum; j++)
  {
    for(int i = 0 ; i < gridColNum; i++)
    {
      if(pitchSquaretotal[i][j] > 0)
      {
        if(firstValue)
        {
          pitchPoints[0] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);
          pitchPoints[1] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);
          pitchPoints[2] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);
          pitchPoints[3] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);

          firstValueGridY = j;
          firstValue = false;
        }
        else
        {
          if(pitchSquarex[i][j] < pitchPoints[0].x)
            pitchPoints[0] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);

          if(pitchSquarex[i][j] > pitchPoints[1].x)
            pitchPoints[1] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);

          if(firstValueGridY + thresholdHeight > j)
          {
            if(pitchPoints[2].x > pitchSquarex[i][j])
              pitchPoints[2] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);

            if(pitchPoints[3].x < pitchSquarex[i][j])
              pitchPoints[3] = Point(pitchSquarex[i][j],pitchSquarey[i][j]);
          }
        }
      }
    }
  }
  return pitchPoints;
}

/**
    fixes the perpective view to a straight view

    @param pitchPoints the corners of the pitch
    @return the transformation needed to get a straight view
*/
Mat perpective_Transformation(vector<Point2f> pitchPoints)
{
  vector<Point2f> outputPoints(4);
  // bottom left
  //outputPoints[0] = Point2f(915, 930);
  outputPoints[0] = pitchPoints[0];

  // bottom right
  outputPoints[1] = pitchPoints[1];
  // outputPoints[1] = Point2f(3900, 925);

  // top left
  outputPoints[2] = Point2f( pitchPoints[0].x, pitchPoints[2].y);

  // top right
  outputPoints[3] = Point2f(pitchPoints[1].x, pitchPoints[3].y);

  return getPerspectiveTransform( pitchPoints, outputPoints );
}

/**
    Updates the background so we can detect motion on the pitch which will
    more than likely be playerse

    @param frame current frame
    @param croppedPitch the pitch area only
    @param fgMask mask of the pitch
    @param pBackSub background substractor itself
    @param differenceFrame frame holding the motion of the pitch
    @return motion dected on the pitch
*/
Ptr<BackgroundSubtractor> update_Background_Sub(Mat frame, Mat *croppedPitch, Mat *fgMask, Ptr<BackgroundSubtractor> pBackSub,
                           Mat *differenceFrame)
{
  bitwise_and(frame, *croppedPitch, *fgMask);
  pBackSub->apply(*fgMask, *fgMask);
  imwrite("fgmask.jpg", *fgMask);

  *differenceFrame = fgMask->clone();
  float erosion_size = 0.5;
  Mat sizeOfErrosion = getStructuringElement( MORPH_RECT,
                  Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                  Point( erosion_size, erosion_size  ) );

  // Remove the shadow parts and the noise
  // threshold(*differenceFrame, *differenceFrame, 128, 255, THRESH_BINARY);
  // [commented as it removes more than shadows]

  erode( *differenceFrame, *differenceFrame, sizeOfErrosion   );

  for(int i = 0 ; i < 3 ; i++)
     dilate( *differenceFrame, *differenceFrame, sizeOfErrosion );

  return pBackSub;
}

/**
    checks to see if detected contours are people by checking if the
    they're a normal size

    @param boundRect the box around each contour of motion
    @param player_contour array that holds the information of if it is a person
    or not
    @param contours of the motion detected
*/
void check_People(vector<Rect> *boundRect, vector<int> *player_contour, vector<vector<cv::Point>> *contours)
{
  for(int i = 0 ; i < contours->size(); i++)
  {
    (*boundRect)[i] = boundingRect((*contours)[i]);
    if((*boundRect)[i].height > 35
      && (*boundRect)[i].width < (*boundRect)[i].height)
    {
      (*player_contour)[i] = 1;
    }
  }

}

/**
    finds if a player and a point are within a selected distance

    @param players to be checked
    @param currentPlayer the person wanting to be checked
    @param givenDistance the distance you want to check the point and the person
    are within
    @return 1 if the person is, 0 if the person isn't
*/
int within_Distance(vector<Player> *players,
                              Rect boundRect, int currentPlayer, int givenDistance)
{
  float shortest_distance = 99999999;
  Point2f current_midpoint = Point2f(((boundRect.tl().x + boundRect.br().x)/2), (boundRect.tl().y + boundRect.br().y)/2);
  float current_Point = distance((*players)[currentPlayer].midpoint, current_midpoint );
  if(current_Point < givenDistance)
    return 1;
  return 0;
}

/**
    finds the SIFT features of an image

    @param tl top left of the ROI of the image
    @param br bottom right of the ROI of the image
    @param frameGiven frame to find the SIFT (with ROI)
    @return the swift features
*/
Mat NewSIFTFeatures(Point2f tl, Point2f br, Mat frameGiven)
{
  // finding the ROI
  Rect ROI_Rec = Rect(tl, br);
  Mat image_ROI = frameGiven(ROI_Rec);

  // Detect the keypoints using SURF Detector, compute the descriptors
  int minHessian = 40;
  vector<KeyPoint> keypoints;
  Mat descriptors;
  Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
  detector->detectAndCompute(image_ROI, noArray(), keypoints, descriptors);
  return descriptors;
}

/**
    matches the SIFT features stored of every player to the ones
    find in the new iage. Outputs the ID with the closest matches using k-nearest
    neighbour

    @param players data
    @param currentPlayer_Used checks if a player has already been classified
    @param boundRect box around the motions on the pitch
    @param frameGiven frame to find the SIFT (with ROI)
    @param searchArea area to search for players
    @return the ID that is most likely to be the player
*/
int sift_Match(vector<Player> *players,vector<int> *currentPlayer_Used, Rect boundRect, Mat frameGiven, int searchArea)
{
    vector<float> person_Matches(players->size());
    for(int i = 0 ; i < person_Matches.size() ; i++)
      person_Matches[i] = 999;

    Mat bound_Box_Features = NewSIFTFeatures(boundRect.tl(),boundRect.br(), frameGiven);
    Rect ROI_Rec = Rect(boundRect.tl(),boundRect.br());
    Mat image_ROI = frameGiven(ROI_Rec);

    int most_Matches_ID = -1;
    if(!bound_Box_Features.empty())
    {
      // Matching descriptor vectors with a FLANN based matcher
      for(int current_Person = 0 ; current_Person < players->size() ; current_Person++)
      {
        if((*currentPlayer_Used)[current_Person] != 1)
        {
          for(int current_Feature = 0 ; current_Feature < (*players)[current_Person].SIFTFeatures.size() ; current_Feature++)
          {
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            vector<vector<DMatch>> knn_matches;
            matcher->knnMatch((*players)[current_Person].SIFTFeatures[current_Feature], bound_Box_Features, knn_matches, 1);

            //-- Filter matches using the threshold
            float ratio_thresh = 700;
            for (size_t i = 0; i < knn_matches.size(); i++)
            {
              if (knn_matches[i][0].distance < ratio_thresh)
              {
                  if(knn_matches[i][0].distance < person_Matches[current_Person])
                    person_Matches[current_Person] = (float)knn_matches[i][0].distance;
              }
            }
          }
        }
      }
    }

    float Most_Matches= 99999;
    for(int current_Person = 0 ; current_Person < players->size() ; current_Person++)
    {
      if(Most_Matches > (person_Matches[current_Person])
         && within_Distance(players, boundRect, current_Person, searchArea)== 1
         && person_Matches[current_Person] > 0
         && (*currentPlayer_Used)[current_Person] != 1)
      {
        Most_Matches = person_Matches[current_Person];
        most_Matches_ID = current_Person;
      }
    }
    return most_Matches_ID;
}

/**
    adds all the data needed for a new player

    @param players data
    @param currentPlayer_Used checks if a player has already been classified
    @param boundRect box around the motions on the pitch
    @param numberOfPlayers that currently exist
    @param i bounding box for this new player to find the sift features
    @param blankFrame current frame
*/
void new_Player(vector<Player> *players, vector<int> *currentPlayer_Used,
                            vector<Rect> *boundRect, int **numberOfPlayers, int i, Mat blankFrame)
{
  players->push_back(Player());
  (*currentPlayer_Used)[**numberOfPlayers] = 1;
  (*players)[**numberOfPlayers].id = **numberOfPlayers;
  (*players)[**numberOfPlayers].br = (*boundRect)[i].br();
  (*players)[**numberOfPlayers].tl = (*boundRect)[i].tl();
  (*players)[**numberOfPlayers].midpoint = Point2f((((*boundRect)[i].tl().x + (*boundRect)[i].br().x)/2),
  ((*boundRect)[i].tl().y + (*boundRect)[i].br().y)/2);
  (*players)[**numberOfPlayers].colour = Scalar((100*random())%255,(100*random())%255,(100*random())%255);

  Mat descriptor = NewSIFTFeatures((*players)[**numberOfPlayers].tl, (*players)[**numberOfPlayers].br, blankFrame);
  if(!descriptor.empty())
  {
      (*players)[**numberOfPlayers].SIFTFeatures.push_back(descriptor);
  }
  **numberOfPlayers = **numberOfPlayers + 1;
}

/**
    Updates player information once player has been found

    @param players data
    @param currentPlayer_Used checks if a player has already been classified
    @param boundRect box around the motions on the pitch
    @param numberOfPlayers number of current playerse
    @param i bounding box for this new player to find the sift features
    @param currentPlayer the currentPlayer you want to update the stats for
    @param blankFrame frame used to update SIFT features
*/
void player_Identify(vector<Player> *players, vector<int> *currentPlayer_Used,
                            vector<Rect> *boundRect, int **numberOfPlayers, int i, int currentPlayer, Mat blankFrame, Mat lambda, float pixelDistance)
{
  // working out data travelled
  Point2f current_midpoint = Point2f((((*boundRect)[i].tl().x +
  (*boundRect)[i].br().x)/2), ((*boundRect)[i].tl().y + (*boundRect)[i].br().y)/2);

  Point2f oldPoint = (*players)[currentPlayer].midpoint;
  (*players)[currentPlayer].midpoint = current_midpoint ;
  // working out the distance used using with pixel unit
  vector<cv::Point2f> untransformedPoints;
  vector<cv::Point2f> transformedPoints;
  untransformedPoints.push_back((*players)[currentPlayer].midpoint);
  untransformedPoints.push_back(oldPoint);
  perspectiveTransform(untransformedPoints, transformedPoints, lambda);

  (*players)[currentPlayer].distanceMoved += pixelDistance*distance(transformedPoints[0], transformedPoints[1]);
  (*currentPlayer_Used)[currentPlayer] = 1;
  (*players)[currentPlayer].br = (*boundRect)[i].br();
  (*players)[currentPlayer].tl = (*boundRect)[i].tl();

  // Add new SIFT pages
  Mat descriptor = NewSIFTFeatures((*players)[currentPlayer].tl, (*players)[currentPlayer].br, blankFrame);
  if(!descriptor.empty())
  {
    if((*players)[currentPlayer].SIFTFeatures.size() < 20)
      (*players)[currentPlayer].SIFTFeatures.push_back(descriptor);
    else
    {
      (*players)[currentPlayer].SIFTFeatures[random() % 20] = descriptor;
    }
  }
}

/**
    works to try to update all the player information

    @param players data
    @param contours the contours of motion
    @param boundRect box around the motions on the pitch
    @param player_contour vector stating if a contour is a person or not
    @param frame current frame
    @param finalImage view changed frame
    @param numberOfPlayers number of players known
    @param lambda used to do the perpective_Transformation
    @param pixelDistance the unit for the pixels
    @return the updated player stats
*/
vector<Player> player_Location(vector<Player> players, vector<vector<cv::Point>> contours,
                    vector<Rect> boundRect, vector<int> player_contour, Mat *frame, Mat *finalImage, int *numberOfPlayers, Mat lambda, float pixelDistance)
{
  vector<int> currentPlayer_Used(players.size() + contours.size());

  Mat blankFrame = frame->clone();
  // checking if there has been a detected player yet
  int firstTime = 1;
  if(players.size() > 0)
    firstTime = 0;

  for(int i = 0 ; i < contours.size(); i++)
  {
    if(player_contour[i] == 1)
    {
      if(firstTime == 1 )
      {
        new_Player(&players, &currentPlayer_Used,
                               &boundRect, &numberOfPlayers, i, blankFrame);
      }
      else
      {
        // if greater than -1 then a player has been identified, if not then no one has been
        int lost = sift_Match(&players, &currentPlayer_Used, boundRect[i], blankFrame, 60);
        if(lost > -1)
        {
          player_Identify(&players, &currentPlayer_Used,
                             &boundRect, &numberOfPlayers, i, lost, blankFrame, lambda, pixelDistance);
        }
        else
        {
          int lost = sift_Match(&players, &currentPlayer_Used, boundRect[i], blankFrame, 120);
          if(lost > -1)
          {
            player_Identify(&players, &currentPlayer_Used,
                                   &boundRect, &numberOfPlayers, i, lost, blankFrame, lambda, pixelDistance);
          }
        }
      }
    }
  }
  return players;
}

/**
    prints all the stats

    @param players data
    @param frame current frame
    @param finalImage view changed frame
    @return stats page
*/
Mat stats_Page(vector<Player> players, Mat *frame, Mat *finalImage)
{
  Mat statImage(1080,300,CV_8UC3,Scalar(0,0,0));

  // enter the player ID you wish to track (0 for all players)
  int selected_player = 0;

  // writing player stats
  int overZero = 0;
  if(selected_player == 0)
  {
    for(int i = 0 ; i < players.size() ; i++)
    {
        putText(statImage,"Player: " + to_string(players[i].id), Point(20,20 + (overZero*60)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(30,200,255), 2.0);
        putText(statImage,"Current Position: (" + to_string(int(players[i].midpoint.x)) + "," + to_string(int(players[i].midpoint.y)) + ")", Point(20,35 + (overZero*60)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
        putText(statImage,"Distance: " + to_string(players[i].distanceMoved), Point(20,50 + (overZero*60)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
        overZero++;
        string labelString = "" + to_string((players)[i].id);
        putText(*frame, labelString, (players)[i].br, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        putText(*finalImage, labelString, (players)[i].br, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        rectangle(*frame, (players)[i].tl, (players)[i].br,  (players)[i].colour , 3, 8, 0);
        rectangle(*finalImage, (players)[i].tl, (players)[i].br,  (players)[i].colour , 3, 8, 0);
    }
  }
  else
  {
    string labelString = "" + to_string((players)[selected_player].id);
    putText(*frame, labelString, (players)[selected_player].br, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    putText(*finalImage, labelString, (players)[selected_player].br, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    rectangle(*frame, (players)[selected_player].tl, (players)[selected_player].br,  (players)[selected_player].colour , 2, 8, 0);
    rectangle(*finalImage, (players)[selected_player].tl, (players)[selected_player].br,  (players)[selected_player].colour , 2, 8, 0);
    putText(statImage,"Player: " + to_string(players[selected_player].id), Point(20,20 + (overZero*60)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(30,200,255), 2.0);
    putText(statImage,"Current Position: (" + to_string(int(players[selected_player].midpoint.x)) + "," + to_string(int(players[selected_player].midpoint.y)) + ")", Point(20,35 + (overZero*60)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
    putText(statImage,"Distance: " + to_string(players[selected_player].distanceMoved), Point(20,50 + (overZero*60)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
  }
  return statImage;
}

/**
    display the views

    @param players data
    @param frame current frame
    @param finalImage view changed frame
    @param differenceFrame the motion detected
*/
void display_Views(vector<Player> players, Mat frame,Mat finalImage,Mat differenceFrame)
{
  Mat statsFrame = stats_Page(players, &frame, &finalImage);
  resize(frame, frame, Size(1920, 1080));
  resize(finalImage, finalImage, Size(1920, 1080));
  imshow("frame", frame );
  imshow("stats", statsFrame);
}


/**
    prints a straight line onto command prompt
*/
void commandLineStraightLine()
{
  for(int i = 0; i< 25 ; i++)
    printf("-");
  printf("\n");
}

/**
    estimates the carmera parameters to try to fix barrel distortion

    @param frame current frame
    @return undistorted frame
*/
Mat undistortFrame(Mat frame)
{
  Mat cameraMatrix, distCoeffs, undistortedFramed;
  // real world position
  vector<vector<Point3f>> objectPoints(1);
  objectPoints = curved_Lines(objectPoints);

  // pixel position
  vector<vector<Point2f>> imagePoints(1);
  imagePoints = equation_of_Straight_Line(imagePoints);

  vector<cv::Mat> rvecs, tvecs;
  calibrateCamera(objectPoints, imagePoints, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);;
  undistort(frame, undistortedFramed, cameraMatrix, distCoeffs);
  return undistortedFramed;
}

/**
    Finds actual distance between each pixel by doing actual distance / num of
    pixels in that distance (6 yard box). Current hard coded into the program.

    @param croppedPitch used to find the hard coded distance position
    @param lambda used to transform the frame to get the right transformation
    @return pixel distance unit
*/
float pixelUnits(Mat croppedPitch, Mat lambda)
{
  // warpPerspective(croppedPitch,croppedPitch,lambda,croppedPitch.size());
  // circle(croppedPitch, Point(3760,750), 20, Scalar(255, 5, 255), 1, 8, 0);
  // circle(croppedPitch, Point(3910,750), 20, Scalar(255, 5, 255), 1, 8, 0);
  // resize(croppedPitch, croppedPitch, Size(1920, 1080));
  // imshow("croppedPitch", croppedPitch);
  float pixelSixYards = distance(Point(3760,750), Point(3910,750));
  return 5.4864 / pixelSixYards;
}

/**
    method to conduct the football tracking
    @param video input
*/
void backgroundSub(VideoCapture cap)
{
  // variables
  Mat frame; // the current frame
  int times = 0;
  int numberOfPlayers = 0;
  Mat differenceFrame;
  Mat finalImage;
  vector<Player> players;
  Mat greenOnly;
  Mat finalLines;
  Mat fgMask;
  Ptr<BackgroundSubtractor> pBackSub;
  Mat lambda( 2, 4, CV_32FC1 );
  Mat croppedPitch;
  float pixelDistance;
  int currentFrameIndex = 0;

  // array to hold the first 20 frames
  vector<Mat> holdingFrames;

  // variable used to check if the first frame to load in is reasdy
  bool firstFrameReady = false;
  if(times == 0)
  {
    commandLineStraightLine();
    printf("Football tracking program\n");
    commandLineStraightLine();
  }
  while(1)
  {
    // the current frame
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    // count for how many frames have been run
    times++;

    // fixing camera distortion
    frame = undistortFrame(frame);
    holdingFrames.push_back(frame.clone());

    // capturing the first few freames to run in inverse to load the
    // background sub
    int createTimes = 6;
    if(times < createTimes)
      printf("Percentage loading frames: %f\n", (double)((double)(times-1) / createTimes)*100);

    if(times == createTimes)
    {
      // create the background model
      pBackSub = createBackgroundSubtractorMOG2();

      // finding green pixels of image
      Mat greenOnly = green_pixels_of_image(&frame);
      printf("Found Green Pixels\n");

      // finding the largest area
      croppedPitch = largest_contour(&greenOnly);
      printf("Found largest green area of the image\n");

      // find the Lines around the pitch
      Mat linesAroundPitch = find_Lines_Around_Pitch(&frame, &croppedPitch);
      printf("Found Lines of the pitch\n");

      // finding corners of the pitch
      vector<Point2f> pitchPoints =  find_corners_Of_Pitch(&linesAroundPitch);
      printf("Found corners of the pitch\n");

      // finding perpective transformation
      lambda = perpective_Transformation(pitchPoints);
      printf("Found perpective transformation\n");
      cvtColor(croppedPitch, croppedPitch, CV_BGR2HSV);

      // finding the pixel distance units for each pixel
      pixelDistance = pixelUnits(croppedPitch, lambda);

      printf("Found the pixel units\n");
      firstFrameReady = true;
    }

    if(firstFrameReady == true)
    {
      if(currentFrameIndex == 0)
      {
        for(int i = createTimes - 1; i >= 0 ; i--)
        {
          printf("Percentage to create background: %f\n", (double)((double)(createTimes -i) / createTimes)*100);
          pBackSub = update_Background_Sub(holdingFrames[i], &croppedPitch, &fgMask, pBackSub, &differenceFrame);
        }
      }
      frame = holdingFrames[currentFrameIndex];

      // blurs frame to reduce noise
      GaussianBlur(frame, frame, Size(3, 3), 0);

      //update the background model to new frames
      if(currentFrameIndex != 0)
        pBackSub = update_Background_Sub(frame, &croppedPitch, &fgMask, pBackSub, &differenceFrame);

      Mat current_Pitch = fgMask.clone();
      bitwise_and(frame, frame,current_Pitch, differenceFrame);
      cvtColor(current_Pitch, current_Pitch, CV_BGR2GRAY);

      // find the players on the pitch
      vector<vector<Point>> contours;
      findContours(current_Pitch, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

      // checking if contours are people and assigning a bounding box
      vector<Rect> boundRect(contours.size());
      vector<int> player_contour(contours.size());
      check_People(&boundRect, &player_contour, &contours);

      // player identifitcation and location
      players = player_Location(players, contours, boundRect, player_contour, &frame, &finalImage, &numberOfPlayers, lambda, pixelDistance);

      // one that has been used
      finalImage = frame.clone();
      warpPerspective(finalImage,finalImage,lambda,finalImage.size());
      display_Views(players, frame, finalImage, differenceFrame);
      currentFrameIndex++;
    }
    // Press  ESC on keyboard to  exit
    char c = (char)waitKey(1);
    if( c == 27 )
      break;
      }
}

int main(int argc, char * argv[])
{
  // setting up video input
  VideoCapture cap;
  // Check if camera opened successfully
  cap = setUpInput(argv, cap);
  if(!cap.isOpened())
  {
    cout << "Error opening video stream" << endl;
    return -1;
  }

  backgroundSub(cap);
  return 0;
  }
