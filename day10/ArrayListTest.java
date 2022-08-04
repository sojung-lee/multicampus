package day10;
import java.util.*;
/*ArrayList 는 java.util.List계열
 * - Vector와 기능은 동일
 * - 단 벡터는 멀티 스레드가 동작하는 환경일 때 동기화가 구현되어 있어
 * 		조금이라도 먼저 저장한 객체가 순차적으로 저장되는 반면,
 *   ArrayList는 동기화가 구현되어 있지 않음
 *   
 *LinkedList : List계열
 *   ArrayList는 순차적 데이터 저장할 때 적합. 중간에 삽입 또는 삭제 ==> 적합하지 않음
 *      
 * */
public class ArrayListTest {

	public static void main(String[] args) {
		ArrayList<String> arrList=new ArrayList<>();
		//Object유형을 저장. 저장 영역을 자동으로 확대한다.
		arrList.add("하이");
		arrList.add("반가워요");
		arrList.add("^^");
		System.out.println("arrList.size(): "+arrList.size());
		
		//Iterator<E> iterator() 이용해서 arrList에 저장된 요소들 한꺼번에 출력하기
		Iterator<String> it=arrList.iterator();
		while(it.hasNext()) {
			String s=it.next();
			System.out.println(s);
		}
		
		String s2=arrList.get(2);
		System.out.println(s2);
		
		List<Integer> arrList2=Arrays.asList(40,10,20,5);
		
		//for, get이용해서 출력하기
		for(int i=0;i<arrList2.size();i++) {
			System.out.println(arrList2.get(i));
		}
		
		//Collections.sort() 메서드를 이용해서 정렬
		//Collections.sort(arrList2);//오름차순 정렬
		Collections.sort(arrList2, Collections.reverseOrder());//내림차순 정렬
		
		System.out.println("---정렬 이후-----------");
		for(Integer val: arrList2) {
			System.out.println(val);
		}
		//arrList2.remove(0);//에러 발생
		arrList.remove(0);//삭제 가능
		System.out.println("---0번째 삭제 이후-----------");
		for(String val: arrList) {
			System.out.println(val);
		}
		//List<E> arr=new ArrayList<E>(); ==> 동적인 컬렉션 생성 (데이터 추가,삭제 가능함)
		//List<E> arr=Arrays.asList(E,E,E...);=> 정적인 컬렉션 생성 (데이터 추가, 삭제 불가능함)
		
		

	}

}
