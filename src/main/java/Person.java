

import java.util.List;

public class Person {
 private String person_id;
 private List<String> pose_keypoints_2d;   
 private List<String> face_keypoints_2d;
 private List<String> hand_left_keypoints_2d;
 private List<String> hand_right_keypoints_2d;
 private List<String> pose_keypoints_3d;
 private List<String> face_keypoints_3d;
 private List<String> hand_left_keypoints_3d;
 private List<String> hand_right_keypoints_3d;

 
public String getPerson_id() {
    return person_id;
}
public void setPerson_id(String person_id) {
    this.person_id = person_id;
}
public List<String> getPose_keypoints_2d() {
    return pose_keypoints_2d;
}
public void setPose_keypoints_2d(List<String> pose_keypoints_2d) {
    this.pose_keypoints_2d = pose_keypoints_2d;
}
public List<String> getFace_keypoints_2d() {
    return face_keypoints_2d;
}
public void setFace_keypoints_2d(List<String> face_keypoints_2d) {
    this.face_keypoints_2d = face_keypoints_2d;
}
public List<String> getHand_left_keypoints_2d() {
    return hand_left_keypoints_2d;
}
public void setHand_left_keypoints_2d(List<String> hand_left_keypoints_2d) {
    this.hand_left_keypoints_2d = hand_left_keypoints_2d;
}
public List<String> getHand_right_keypoints_2d() {
    return hand_right_keypoints_2d;
}
public void setHand_right_keypoints_2d(List<String> hand_right_keypoints_2d) {
    this.hand_right_keypoints_2d = hand_right_keypoints_2d;
}
public List<String> getPose_keypoints_3d() {
    return pose_keypoints_3d;
}
public void setPose_keypoints_3d(List<String> pose_keypoints_3d) {
    this.pose_keypoints_3d = pose_keypoints_3d;
}
public List<String> getFace_keypoints_3d() {
    return face_keypoints_3d;
}
public void setFace_keypoints_3d(List<String> face_keypoints_3d) {
    this.face_keypoints_3d = face_keypoints_3d;
}
public List<String> getHand_left_keypoints_3d() {
    return hand_left_keypoints_3d;
}
public void setHand_left_keypoints_3d(List<String> hand_left_keypoints_3d) {
    this.hand_left_keypoints_3d = hand_left_keypoints_3d;
}
public List<String> getHand_right_keypoints_3d() {
    return hand_right_keypoints_3d;
}
public void setHand_right_keypoints_3d(List<String> hand_right_keypoints_3d) {
    this.hand_right_keypoints_3d = hand_right_keypoints_3d;
}
 


}

