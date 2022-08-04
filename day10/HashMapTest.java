package day10;
import java.util.*;
import java.io.*;
//HashMap
// - Map 계열
// - 무순서 (입력순서와 출력순서가 다를 수 있음)
// - Key값을 Set으로 관리한다
public class HashMapTest {
	HashMap<String, String> map=new HashMap<>();
	
	public void join() {		
		map.put("Hong", "123");
		map.put("Lee", "abc");
		map.put("Admin", "tiger");
		System.out.println(">>회원 가입 완료 [회원수: "+map.size()+"명]<<");
	}//------------------------
	
	public void loginCheck() {
		Console console=System.console();
		//이클립스에서 테스트하면 x => console이 null이 되므로
		//도스창 열어 테스트 하기
		while(true) {
			System.out.print("아이디: ");
			String userid=console.readLine();
			//System.out.println("userid: "+userid);
			
			System.out.print("비밀번호: ");
			char[] pwd=console.readPassword();
			//System.out.println("pwd: "+new String(pwd));
			//1) 입력한 아이디가 map에 저장되어있는지 체크
			//	  1_1) 없다면 =>"입력오류!. 회원이 아닙니다"
			//	  1_2) 있다면 
			//			<a> 비밀번호가 일치하는 경우 =>"~~님 환영합니다"
			//			<b> 일치하지 않는 경우 => "비밀번호가 일치하지 않아요"
			if (!map.containsKey(userid)) {
				System.out.println("입력오류! 회원이 아닙니다.");
			} else {
				String realPwd=map.get(userid);
				//== : 기본자료형: 값을 비교
				//	   참조유형: 주소값을 비교
				//boolean equals(Object o) : String의 경우 문자열의 내용이 같으면 true반환
				if (realPwd.equals(new String(pwd))) {
					System.out.println(userid + "님 환영합니다.");
					break;
				} else {
					System.out.println("비밀번호가 일치하지 않습니다.");
					continue;
				}
			}
		}//while------------
		
	}//------------------------

	public static void main(String[] args) {
		//join()을 호출하세요
		HashMapTest hm=new HashMapTest();
		hm.join();
		hm.loginCheck();
	}

}
