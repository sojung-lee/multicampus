package day13;

import java.io.IOException;

/* System.in  : InputStream타입
- 1byte 기반 스트림
- 노드 스트림: 키보드와 노드 연결하는 스트림
- public int read() :
  바이트 입력을 리턴하고 입력이 종료(Ctrl+C 또는
   Ctrl+D)되면 -1을 리턴한다.
*/
public class InputStreamTest {

	public static void main(String[] args) {
		int input=0;
		int count=0;
		System.out.println("입력하세요=>");
		try {
			while(true) {
				input=System.in.read();
				//키보드 입력
				System.out.println("input: "+input);
				count++;
				if(input=='x') break;
			}//while---
			System.out.println("총 "+count+"바이트 입력받음");
		}catch(IOException e) {
			e.printStackTrace();
		}

	}

}
