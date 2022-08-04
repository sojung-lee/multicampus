package day13;
import java.io.*;
//System.in : 키보드와 노드 연결
//System.out: 콘솔과 노드 연결
public class InputStreamTest3 {

	public static void main(String[] args)  
	throws IOException
	{
		int n=0;
		int cnt=0;
		System.out.println("입력하세요=>");
		//데이터를 달걀판(배열)에 담아서 읽고 써보자
		byte[] arr=new byte[6];//달걀판
		int total=0;
		while((n=System.in.read(arr))!= -1) {
			//배열arr에 데이터가 담긴다
			//n: 달걀갯수 (읽은 바이트수)
			//System.out.write(n);[x]
			System.out.write(arr, 0, n);
			System.out.flush();//스트림에 남아있는 데이터를 밀어내기 해줌
			cnt++;//반복문 횟수
			total+=n;//입력받은 바이트 수
		}
		System.out.println(total+"bytes 읽음");
		
		System.in.close();
		System.out.close();

	}

}
